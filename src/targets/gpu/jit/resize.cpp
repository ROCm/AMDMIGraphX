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
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/transform_view.hpp>
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
        ${resize_func}<${coord_transform}, ${nearest_op}>(input, out, ${scales});
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
        return "make_array<float>(" + to_string_range(views::transform(scales, MIGRAPHX_LIFT(to_hex_float))) + ")";
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        if(inputs.size() != 2)
            MIGRAPHX_THROW("GPU resize: Incorrect arguments");

        const auto& in_lens = inputs.front().lens();

        // Compute scales from shapes
        std::vector<float> scales;
        const auto& out_lens = inputs.back().lens();
        scales.resize(in_lens.size());
        std::transform(in_lens.begin(),
                       in_lens.end(),
                       out_lens.begin(),
                       scales.begin(),
                       [](float in, float out) { return out / in; });

        hip_compile_options options;
        options.set_launch_params(v, compute_global_for(ctx, inputs.back().elements(), 1024));
        options.output      = inputs.back();
        options.inputs      = inputs;
        options.kernel_name = "resize";

        // Get mode (nearest or linear)
        std::string resize_func = "resize_" + v.get("mode", "nearest");

        // Get coordinate transformation mode
        std::string coord_transform =
            "coord_transform_" + v.get("coordinate_transformation_mode", "half_pixel");

        // Get nearest mode (only used for nearest interpolation)
        std::string nearest_op = "nearest_" + v.get("nearest_mode", "floor");

        auto src = interpolate_string(resize_kernel,
                                      {{"coord_transform", coord_transform},
                                       {"nearest_op", nearest_op},
                                       {"scales", scales_to_string(scales)},
                                       {"resize_func", resize_func}});

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
