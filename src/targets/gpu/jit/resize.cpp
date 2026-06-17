/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/transform_view.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/float_equal.hpp>
#include <sstream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const resize_kernel = R"__migraphx__(
#include <migraphx/kernels/resize.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void resize(void* in_data, void* output)
{
    make_tensors()(in_data, output)([](auto input, auto out) {
        ${in_vectorize}(input)([&](auto inv) {
            ${vectorize}(out)([&](auto outv) {
                ${resize_func}<${coord_transform}, ${nearest_op}, ${axis}>(
                    inv, out, outv, ${scales}${cubic_coeff_arg});
            });
        });
    });
}

}

} // namespace migraphx

)__migraphx__";

struct resize_compiler : compiler<resize_compiler>
{
    std::vector<std::string> names() const { return {"resize"}; }

    static std::string scales_to_string(const std::vector<float>& scales)
    {
        return "make_array<float>(" +
               to_string_range(views::transform(scales, MIGRAPHX_LIFT(to_hex_float))) + ")";
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        if(inputs.size() != 2)
            MIGRAPHX_THROW("GPU resize: Incorrect arguments");

        std::vector<int64_t> permutation;
        hip_compile_options options;
        options.output         = inputs.back();
        options.inputs         = inputs;
        options.kernel_name    = "resize";
        options.virtual_inputs = normalize_permutation(inputs, &permutation);

        std::string mode       = v.get("mode", "nearest");
        std::string coord_mode = v.get("coordinate_transformation_mode", "half_pixel");

        // The output store is vectorized along the fastest axis. Pick the size/axis with the
        // shared vectorize helper and launch one thread per vectorized element. Resize is faster
        // with a width-4 store than the half2 default, so the candidate sizes are given explicitly.
        auto vec = gen::vectorize::elements(gen::find_fast_axis(options.virtual_inputs.back()),
                                            {options.virtual_inputs.back()},
                                            {4, 2});
        options.set_launch_params(
            v, compute_global_for(ctx, inputs.back().elements() / vec.size, 1024));

        // Compute scales from shapes
        const auto& in_lens       = options.virtual_inputs.front().lens();
        const auto& in_strides    = options.virtual_inputs.front().strides();
        const auto& out_lens      = options.virtual_inputs.back().lens();
        std::vector<float> scales = v.at("scales").to_vector<float>();
        if(scales.size() != in_lens.size())
        {
            scales.resize(in_lens.size());
            std::transform(in_lens.begin(),
                           in_lens.end(),
                           out_lens.begin(),
                           scales.begin(),
                           [](float in, float out) { return out / in; });
        }
        else
        {
            scales = reorder_dims(scales, permutation);
        }

        // The input is gathered, so it can share the output's vectorization only when the fast
        // axis is a genuine pass-through (its input index equals its output index): equal length,
        // unit scale, contiguous, and an identity-at-unit-scale coordinate transform. Otherwise
        // the input transformer is the identity and the input is gathered scalar-wise.
        //
        // The mode list below is exactly the coord_transform_* function objects in
        // kernels/resize.hpp whose operator() maps idx -> idx at unit scale (every mode except
        // tf_half_pixel_for_nn); keep it in sync if a coordinate transform is added or changed. An
        // omission is safe (it just falls back to a scalar gather), but listing a non-identity mode
        // would be incorrect.
        const bool axis_passthrough =
            in_lens[vec.axis] == out_lens[vec.axis] and in_strides[vec.axis] == 1 and
            float_equal(scales[vec.axis], 1.0f) and
            contains({"half_pixel", "pytorch_half_pixel", "align_corners", "asymmetric"},
                     coord_mode);
        gen::vectorize in_vec =
            axis_passthrough
                ? gen::vectorize::elements(vec.axis, {options.virtual_inputs.front()}, {vec.size})
                : gen::vectorize{1, vec.axis};

        std::string resize_func     = "resize_" + mode;
        std::string coord_transform = "coord_transform_" + coord_mode;

        // Get nearest mode (only used for nearest interpolation)
        std::string nearest_op = "nearest_" + v.get("nearest_mode", "floor");

        // Handle cubic coefficient (only used for cubic mode)
        std::string cubic_coeff_arg;
        if(mode == "cubic")
        {
            float cubic_coeff = v.get("cubic_coeff_a", -0.75f);
            cubic_coeff_arg   = ", " + to_hex_float(cubic_coeff) + "f";
        }

        auto src = interpolate_string(resize_kernel,
                                      {{"coord_transform", coord_transform},
                                       {"nearest_op", nearest_op},
                                       {"scales", scales_to_string(scales)},
                                       {"resize_func", resize_func},
                                       {"vectorize", vec.str()},
                                       {"in_vectorize", in_vec.str()},
                                       {"axis", std::to_string(vec.axis)},
                                       {"cubic_coeff_arg", cubic_coeff_arg}});

        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
