/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

// Version 3: Double precision coord transforms
MIGRAPHX_GLOBAL void ${kernel_name}(void* in_data, void* output)
{
    make_tensors()(in_data, output)([](auto input, auto out) {
        auto settings = make_resize_settings(
            ${coord_transform}{},
            ${nearest_op}{},
            ${scales});
        ${resize_func}(input, out, settings);
    });
}

}

} // namespace migraphx

)__migraphx__";

struct resize_compiler : compiler<resize_compiler>
{
    std::vector<std::string> names() const { return {"resize"}; }

    static std::string get_coord_transform(const std::string& coord_mode)
    {
        if(coord_mode == "half_pixel")
            return "coord_transform_half_pixel";
        if(coord_mode == "pytorch_half_pixel")
            return "coord_transform_pytorch_half_pixel";
        if(coord_mode == "align_corners")
            return "coord_transform_align_corners";
        if(coord_mode == "asymmetric")
            return "coord_transform_asymmetric";
        if(coord_mode == "tf_half_pixel_for_nn")
            return "coord_transform_tf_half_pixel_for_nn";
        // Default to half_pixel
        return "coord_transform_half_pixel";
    }

    static std::string get_nearest_op(const std::string& nearest_mode)
    {
        if(nearest_mode == "floor")
            return "nearest_floor";
        if(nearest_mode == "ceil")
            return "nearest_ceil";
        if(nearest_mode == "round_prefer_floor")
            return "nearest_round_prefer_floor";
        if(nearest_mode == "round_prefer_ceil")
            return "nearest_round_prefer_ceil";
        // Default to floor
        return "nearest_floor";
    }

    static std::string scales_to_string(const std::vector<float>& scales)
    {
        std::string result = "make_array<float>(";
        for(size_t i = 0; i < scales.size(); ++i)
        {
            if(i > 0)
                result += ", ";
            // Use hexfloat format to preserve exact float representation
            std::ostringstream oss;
            oss << std::hexfloat << scales[i] << "f";
            result += oss.str();
        }
        result += ")";
        return result;
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        if(inputs.size() != 2)
            MIGRAPHX_THROW("GPU resize: Incorrect arguments");

        const auto& in_lens = inputs.front().lens();

        // Compute scales from value or from shapes
        std::vector<float> scales;
        if(v.contains("scales") and not v.at("scales").empty())
        {
            scales = v.at("scales").to_vector<float>();
        }
        else if(v.contains("sizes") and not v.at("sizes").empty())
        {
            auto out_lens = v.at("sizes").to_vector<size_t>();
            scales.resize(in_lens.size());
            for(size_t i = 0; i < in_lens.size(); ++i)
            {
                scales[i] = static_cast<float>(out_lens[i]) / static_cast<float>(in_lens[i]);
            }
        }
        else
        {
            // No scales or sizes - compute from shapes
            const auto& out_lens = inputs.back().lens();
            scales.resize(in_lens.size());
            for(size_t i = 0; i < in_lens.size(); ++i)
            {
                scales[i] = static_cast<float>(out_lens[i]) / static_cast<float>(in_lens[i]);
            }
        }

        hip_compile_options options;
        options.set_launch_params(v, compute_global_for(ctx, inputs.back().elements(), 1024));
        options.output      = inputs.back();
        options.inputs      = inputs;

        // Include input and output shapes in kernel name plus version to prevent incorrect caching
        // when the same operation is used with different configurations
        // v3: Using double precision for coordinate transforms
        std::string kernel_name = "resize_v3";
        for(const auto& dim : inputs.front().lens())
            kernel_name += "_" + std::to_string(dim);
        kernel_name += "_to";
        for(const auto& dim : inputs.back().lens())
            kernel_name += "_" + std::to_string(dim);
        options.kernel_name = kernel_name;

        // Get mode (nearest or linear)
        std::string mode        = v.get("mode", "nearest");
        std::string resize_func = (mode == "linear") ? "resize_linear" : "resize_nearest";

        // Get coordinate transformation mode
        std::string coord_mode      = v.get("coordinate_transformation_mode", "half_pixel");
        std::string coord_transform = get_coord_transform(coord_mode);

        // Get nearest mode (only used for nearest interpolation)
        std::string nearest_mode = v.get("nearest_mode", "floor");
        std::string nearest_op   = get_nearest_op(nearest_mode);

        auto src = interpolate_string(resize_kernel,
                                      {{"kernel_name", options.kernel_name},
                                       {"coord_transform", coord_transform},
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
