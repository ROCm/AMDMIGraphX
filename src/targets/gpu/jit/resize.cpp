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
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/op/resize.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/algorithm.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const resize_kernel = R"__migraphx__(
#include <migraphx/kernels/resize.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void ${kernel}(${params})
{
    auto scales = ${scales};
    make_tensors()(${args})([&](auto input${maybe_scales}, auto output) {
        resize_linear_asymmetric(input, output, scales);
    });
}

}

} // namespace migraphx

)__migraphx__";

struct resize_compiler : compiler<resize_compiler>
{
    std::vector<std::string> names() const { return {"resize"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        if(inputs.size() < 2)
            MIGRAPHX_THROW("GPU resize: missing output argument");

        if(std::any_of(inputs.begin(), inputs.end(), [](const auto& s) { return s.dynamic(); }))
            MIGRAPHX_THROW("GPU resize: dynamic shapes not supported in JIT kernel");

        auto mode = v.get("mode", std::string{"nearest"});
        if(mode != "linear")
            MIGRAPHX_THROW("GPU resize: only linear mode is supported in JIT kernel");

        auto coord_mode = v.get("coordinate_transformation_mode", std::string{"half_pixel"});
        if(coord_mode != "asymmetric")
            MIGRAPHX_THROW(
                "GPU resize: only asymmetric coordinate_transformation_mode is supported");

        const auto& input_shape = inputs.front();
        auto scales              = v.get("scales", std::vector<float>{});
        auto sizes               = v.get("sizes", std::vector<std::size_t>{});
        if(scales.empty())
        {
            if(sizes.empty())
                MIGRAPHX_THROW("GPU resize: scales or sizes must be provided");
            if(sizes.size() != input_shape.lens().size())
                MIGRAPHX_THROW("GPU resize: sizes rank does not match input rank");
            scales.resize(sizes.size());
            std::transform(sizes.begin(),
                           sizes.end(),
                           input_shape.lens().begin(),
                           scales.begin(),
                           [](std::size_t out_len, std::size_t in_len) {
                               return (in_len == 0) ? 1.0f
                                                    : static_cast<float>(out_len) / in_len;
                           });
        }

        if(scales.size() != input_shape.lens().size())
            MIGRAPHX_THROW("GPU resize: scales rank does not match input rank");

        hip_compile_options options;
        options.inputs      = inputs;
        options.output      = inputs.back();
        options.kernel_name = "resize_kernel";
        options.set_launch_params(v, compute_global_for(ctx, options.output.elements()));

        std::string scales_value =
            "migraphx::array<float, " + std::to_string(scales.size()) + ">{" +
            to_string_range(scales) + "}";

        std::string maybe_scales;
        if(inputs.size() == 3)
            maybe_scales = ", auto";
        else if(inputs.size() != 2)
            MIGRAPHX_THROW("GPU resize: unexpected number of arguments");

        auto src = interpolate_string(resize_kernel,
                                      {{"kernel", options.kernel_name},
                                       {"params", enum_params(inputs.size(), "void* private_p")},
                                       {"args", enum_params(inputs.size(), "private_p")},
                                       {"scales", scales_value},
                                       {"maybe_scales", maybe_scales}});
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
