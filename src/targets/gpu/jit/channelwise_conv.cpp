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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const channelwise_conv_kernel = R"__migraphx__(
#include <migraphx/kernels/channelwise_conv.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void channelwise_conv_kernel(void* x_p, void* w_p, void* y_p)
{
    transform_args(make_tensors(), rotate_last())(x_p, w_p, y_p)([](auto output, auto x, auto w) {
        channelwise_conv<${algo}>(index_ints<${kernel}>{}, output, x, w);
    });
}

}

} // namespace migraphx

)__migraphx__";

struct channelwise_conv_compiler : compiler<channelwise_conv_compiler>
{
    std::vector<std::string> names() const
    {
        return {"gpu::channelwise_conv", "channelwise_conv"};
    }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        auto num_spatial   = v.at("num_spatial").to<std::size_t>();
        const auto& x_s    = inputs.at(0);
        const auto& out_s  = inputs.back();
        options.inputs     = inputs;
        options.output     = out_s;
        options.kernel_name = "channelwise_conv_kernel";
        options.virtual_inputs = inputs;

        auto x_lens = x_s.lens();
        std::vector<std::size_t> kernel_sizes(x_lens.begin() + 2,
                                              x_lens.begin() + 2 + num_spatial);
        std::size_t kernel_total = 1;
        for(auto k : kernel_sizes)
            kernel_total *= k;

        std::string algo       = "reduce::lane";
        std::size_t block_size = 256;

        options.set_launch_params(
            v, compute_global_for(ctx, out_s.elements(), 256), block_size);

        auto src = interpolate_string(channelwise_conv_kernel,
                                      {{"algo", algo},
                                       {"kernel", to_string_range(kernel_sizes)}});

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
