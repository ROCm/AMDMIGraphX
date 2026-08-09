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
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const rope_qk_norm_kernel = R"__migraphx__(
#include <args.hpp>
#include <migraphx/kernels/rope_qk_norm.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void ${kernel}(${params})
{
    transform_args(make_tensors())(${args})([](auto... xs) {
        rope_qk_norm<${num_heads}>(xs..., ${eps}, ${ss_scale});
    });
}

}

} // namespace migraphx

)__migraphx__";

struct rope_qk_norm_compiler : compiler<rope_qk_norm_compiler>
{
    std::vector<std::string> names() const { return {"gpu::rope_qk_norm"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        const auto& out_shape = inputs.back();
        auto nheads           = out_shape.lens().at(0) * out_shape.lens().at(1);

        hip_compile_options options;
        auto local = 64;
        options.set_launch_params(v, nheads * local, local);
        options.inputs      = inputs;
        options.output      = out_shape;
        options.kernel_name = "rope_qk_norm_kernel";

        auto src =
            interpolate_string(rope_qk_norm_kernel,
                               {{"params", enum_params(inputs.size(), "void * private_p")},
                                {"args", enum_params(inputs.size(), "private_p")},
                                {"kernel", options.kernel_name},
                                {"num_heads", std::to_string(v.at("num_heads").to<std::size_t>())},
                                {"eps", std::to_string(v.at("eps").to<float>()) + "f"},
                                {"ss_scale", std::to_string(v.at("ss_scale").to<float>()) + "f"}});
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
