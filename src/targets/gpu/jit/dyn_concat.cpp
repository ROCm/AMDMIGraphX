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
#include <migraphx/gpu/compile_hip.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/sym.hpp>

#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const dyn_concat_kernel_src = R"__migraphx__(
#include <migraphx/kernels/dyn_concat.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void dyn_concat_kernel(${params})
{
    transform_args(make_tensors(), rotate_last<2>())(${args})([](auto output,
                                                                 auto total,
                                                                 auto counts,
                                                                 auto... inputs) {
        dyn_concat<${axis}>(counts, output, total, inputs...);
    });
}

}

} // namespace migraphx
)__migraphx__";

struct dyn_concat_compiler : compiler<dyn_concat_compiler>
{
    std::vector<std::string> names() const { return {"dyn_concat"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        if(inputs.size() < 3)
            MIGRAPHX_THROW("DYN_CONCAT: invalid precompile input list");
        const auto num_inputs = inputs.size() - 2;
        const auto rank       = inputs.at(1).ndim();
        auto axis_value       = v.at("axis").to<int64_t>();
        if(axis_value < 0)
            axis_value += static_cast<int64_t>(rank);
        const auto axis = static_cast<std::size_t>(axis_value);

        hip_compile_options options;
        options.inputs = flatten_tuple_shapes(inputs);
        std::transform(inputs.begin() + 1,
                       inputs.begin() + 1 + num_inputs,
                       options.inputs.begin() + 1,
                       [](const shape& s) {
                           if(not s.dynamic())
                               return s;
                           std::vector<std::size_t> strides;
                           std::transform(s.dyn_strides().begin(),
                                          s.dyn_strides().end(),
                                          std::back_inserter(strides),
                                          [](const auto& stride) {
                                              return sym::to<std::size_t>(
                                                  *sym::fixed_value(stride));
                                          });
                           return shape{s.type(), s.max_lens(), strides};
                       });
        options.output             = inputs.back();
        options.virtual_inputs     = options.inputs;
        options.kernel_name        = "dyn_concat_kernel";
        const auto output_elements = inputs.back().sub_shapes().front().elements();
        const auto block_size      = compute_block_size(
            ctx, std::max<std::size_t>(num_inputs, std::max<std::size_t>(1, output_elements)), 256);
        options.set_launch_params(v, block_size, block_size);

        auto src =
            interpolate_string(dyn_concat_kernel_src,
                               {{"params", enum_params(options.inputs.size(), "void * private_p")},
                                {"args", enum_params(options.inputs.size(), "private_p")},
                                {"axis", std::to_string(axis)}});
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
