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
#include <migraphx/split_sym/analyzer.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

struct analyze_gather : analyzer<analyze_gather>
{
    bool matches(const operation& op) const { return op.name() == "gather"; }

    symbolic_op_info analyze(instruction_ref ins) const
    {
        auto inputs = to_shapes(ins->inputs());
        if(inputs.size() != 2)
            return symbolic_op_info{ins};
        auto axis = normalize_axis(ins->get_operator().to_value().at("axis").to<int64_t>(),
                                   inputs.front().ndim());
        if(not axis.has_value())
            return symbolic_op_info{ins};
        const auto data_dims              = inputs.front().to_symbolic().dyn_dims();
        const bool has_variable_data_axis = is_variable_axis(data_dims.at(*axis));
        if(has_variable_data_axis and
           not contains({shape::int32_type, shape::int64_type}, inputs.back().type()))
            return symbolic_op_info{ins};
        // The freezer normalizes negative indices against the unpadded extent, so extending the
        // data axis cannot change which valid element an index selects.
        auto result =
            analyze_axes(ins, [axis = *axis](std::size_t input, std::size_t current_axis) {
                if(input == 1)
                    return parallel_axis();
                return current_axis == axis ? padded_axis(fill_kind::dont_care, true)
                                            : parallel_axis();
            });
        if(has_variable_data_axis and result.supported)
            result.freezer = freeze;
        return result;
    }

    static instruction_ref freeze(module& m,
                                  instruction_ref source,
                                  const std::vector<instruction_ref>& args,
                                  const std::unordered_map<sym::expr, std::size_t>& values)
    {
        assert(args.size() == 2);
        const auto& source_data_shape = source->inputs().front()->get_shape();
        auto axis = normalize_axis(source->get_operator().to_value().at("axis").to<int64_t>(),
                                   source_data_shape.ndim());
        assert(axis.has_value());
        const auto data_dims = source_data_shape.to_symbolic().dyn_dims();
        const auto& extent   = data_dims.at(*axis);
        if(not is_variable_axis(extent))
            return m.add_instruction(source->get_operator(), args);

        const auto data               = args.front();
        const auto indices            = args.back();
        const auto parameters         = m.get_parameters();
        const auto extent_variables   = sym::find_variables(extent.sym_expr);
        const bool has_runtime_source = any_of(parameters, [&](instruction_ref parameter) {
            const auto& parameter_shape = parameter->get_shape();
            if(not parameter_shape.symbolic())
                return false;
            return any_of(parameter_shape.dyn_dims(), [&](const auto& dimension) {
                auto variables = sym::find_variables(dimension.sym_expr);
                return any_of(variables, [&](const auto& variable) {
                    return contains(extent_variables, variable);
                });
            });
        });
        instruction_ref runtime_extent;
        if(has_runtime_source)
        {
            runtime_extent = m.add_instruction(
                make_op("eval_expr_from_shape",
                        {{"expressions", to_value(std::vector<sym::expr>{extent.sym_expr})}}),
                parameters);
        }
        else
        {
            const auto value = extent.sym_expr.eval_uint(values);
            runtime_extent   = m.add_literal(literal{
                shape{shape::int64_type, {1}}, std::vector<int64_t>{static_cast<int64_t>(value)}});
        }
        if(indices->get_shape().type() != shape::int64_type)
            runtime_extent = m.add_instruction(
                make_op("convert", {{"target_type", indices->get_shape().type()}}), runtime_extent);
        runtime_extent = m.add_instruction(
            make_op("multibroadcast", {{"out_lens", indices->get_shape().lens()}}), runtime_extent);
        auto zero = m.add_literal(
            literal{shape{indices->get_shape().type(), {1}}, std::vector<int64_t>{0}});
        zero = m.add_instruction(
            make_op("multibroadcast", {{"out_lens", indices->get_shape().lens()}}), zero);
        auto negative   = m.add_instruction(make_op("less"), indices, zero);
        auto normalized = m.add_instruction(make_op("add"), indices, runtime_extent);
        normalized      = m.add_instruction(make_op("where"), negative, normalized, indices);
        return m.add_instruction(source->get_operator(), data, normalized);
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
