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
#include <migraphx/stringutils.hpp>
#include <migraphx/sym.hpp>

#include <algorithm>
#include <numeric>

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
    make_tensors()(${args})([](${lambda_params}) {
        auto idx = make_index();
        __shared__ index_int offsets[${num_inputs}];
        if(idx.local == 0)
        {
${offsets}
            total[0] = ${total};
        }
        __syncthreads();
${copies}
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
        if(inputs.size() < 3 or inputs.size() % 2 == 0)
            MIGRAPHX_THROW("DYN_CONCAT: invalid precompile input list");
        const auto num_inputs = (inputs.size() - 1) / 2;
        const auto rank       = inputs.front().ndim();
        auto axis_value       = v.at("axis").to<int64_t>();
        if(axis_value < 0)
            axis_value += static_cast<int64_t>(rank);
        const auto axis = static_cast<std::size_t>(axis_value);

        std::vector<std::size_t> capacities(num_inputs);
        std::transform(inputs.begin(),
                       inputs.begin() + num_inputs,
                       capacities.begin(),
                       [&](const shape& s) { return s.max_lens().at(axis); });
        const auto output_capacity =
            std::accumulate(capacities.begin(), capacities.end(), std::size_t{0});
        const auto max_lens = inputs.front().max_lens();
        const auto outer    = std::accumulate(
            max_lens.begin(), max_lens.begin() + axis, std::size_t{1}, std::multiplies<>{});
        const auto inner = std::accumulate(
            max_lens.begin() + axis + 1, max_lens.end(), std::size_t{1}, std::multiplies<>{});

        std::vector<std::string> lambda_params;
        transform(range(num_inputs), std::back_inserter(lambda_params), [](auto i) {
            return "auto input" + std::to_string(i);
        });
        transform(range(num_inputs), std::back_inserter(lambda_params), [](auto i) {
            return "auto count" + std::to_string(i);
        });
        lambda_params.push_back("auto output");
        lambda_params.push_back("auto total");

        std::vector<std::string> offsets;
        std::vector<std::string> copies;
        transform(range(num_inputs), std::back_inserter(offsets), [&](std::size_t i) {
            if(i == 0)
                return std::string{"            offsets[0] = 0;"};
            return "            offsets[" + std::to_string(i) + "] = offsets[" +
                   std::to_string(i - 1) + "] + " + "dyn_concat_count<" +
                   std::to_string(capacities.at(i - 1)) + ">(count" + std::to_string(i - 1) + ");";
        });
        transform(range(num_inputs), std::back_inserter(copies), [&](std::size_t i) {
            auto count = "dyn_concat_count<" + std::to_string(capacities.at(i)) + ">(count" +
                         std::to_string(i) + ")";
            const auto& input_shape   = inputs.at(i);
            const bool fixed_capacity = not input_shape.dynamic() or
                                        (input_shape.symbolic() and
                                         all_of(input_shape.dyn_strides(), [](const auto& stride) {
                                             return sym::fixed_value(stride).has_value();
                                         }));
            const auto source_capacity = fixed_capacity ? std::to_string(capacities.at(i)) : count;
            return "        dyn_concat_copy<" + std::to_string(outer) + ", " +
                   std::to_string(capacities.at(i)) + ", " + std::to_string(output_capacity) +
                   ", " + std::to_string(inner) + ">(idx, input" + std::to_string(i) + ", " +
                   count + ", " + source_capacity + ", offsets[" + std::to_string(i) +
                   "], output);";
        });
        auto total = "offsets[" + std::to_string(num_inputs - 1) + "] + " + "dyn_concat_count<" +
                     std::to_string(capacities.back()) + ">(count" +
                     std::to_string(num_inputs - 1) + ")";

        hip_compile_options options;
        options.inputs = flatten_tuple_shapes(inputs);
        std::transform(inputs.begin(),
                       inputs.begin() + num_inputs,
                       options.inputs.begin(),
                       [](const shape& s) { return shape{s.type(), s.max_lens()}; });
        options.output         = inputs.back();
        options.virtual_inputs = options.inputs;
        options.kernel_name    = "dyn_concat_kernel";
        const auto block_size =
            compute_block_size(ctx, std::max<std::size_t>(1, output_capacity * outer * inner), 256);
        options.set_launch_params(v, block_size, block_size);

        auto src =
            interpolate_string(dyn_concat_kernel_src,
                               {{"params", enum_params(options.inputs.size(), "void * private_p")},
                                {"args", enum_params(options.inputs.size(), "private_p")},
                                {"lambda_params", join_strings(lambda_params, ", ")},
                                {"num_inputs", std::to_string(num_inputs)},
                                {"offsets", join_strings(offsets, "\n")},
                                {"total", total},
                                {"copies", join_strings(copies, "\n")}});
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
