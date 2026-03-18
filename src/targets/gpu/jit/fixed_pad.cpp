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
#include <migraphx/reduce_dims.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const fixed_pad_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/algorithm.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void fixed_pad_kernel(void* input_p, void* output_p)
{
    auto idx = make_index();
    make_tensors()(input_p, output_p)([&](auto input, auto output) {
        auto input_bounds = input.get_shape().lens;
        idx.global_stride(output.get_shape().elements(), [&](auto i) {
            auto out_idx   = output.get_shape().multi(i);
            bool in_bounds = sequence(out_idx.size(), [&](auto... js) {
                return ((out_idx[js] < input_bounds[js]) and ...);
            });
            output[out_idx] = in_bounds ? input[out_idx] : 0;
        });
    });
}

}

} // namespace migraphx

)__migraphx__";

// NOLINTNEXTLINE
static const char* const fixed_pad_standard_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void fixed_pad_kernel(void* input_p, void* output_p)
{
    auto idx = make_index();
    make_tensors()(input_p, output_p)([&](auto input, auto output) {
        auto ielements = input.get_shape().elements();
        idx.global_stride(output.get_shape().elements(), [&](auto i) {
            output[i] = (i < ielements) ? input[i] : 0;
        });
    });
}

}

} // namespace migraphx

)__migraphx__";

struct fixed_pad_compiler : compiler<fixed_pad_compiler>
{
    std::vector<std::string> names() const { return {"fixed_pad"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        options.inputs      = inputs;
        options.output      = inputs.back();
        options.kernel_name = "fixed_pad_kernel";

        const auto& input_shape  = inputs.front();
        const auto& output_shape = inputs.back();

        bool use_standard_kernel = input_shape.standard() and output_shape.standard();
        if(use_standard_kernel)
        {
            auto ilens            = input_shape.lens();
            auto olens            = output_shape.lens();
            auto [istart, ostart] = std::mismatch(ilens.begin(), ilens.end(), olens.begin());
            use_standard_kernel   = std::equal(istart, ilens.end(), ostart, olens.end());
        }

        options.virtual_inputs = reduce_dims(inputs);
        options.set_launch_params(v, compute_global_for(ctx, output_shape.elements()));

        const char* src = use_standard_kernel ? fixed_pad_standard_kernel : fixed_pad_kernel;
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
