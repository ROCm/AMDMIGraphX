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
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace split_sym {
namespace {

instruction_ref
add_runtime_extent(module& m, const sym::expr& expression, const freezer_context& context)
{
    const auto variables = sym::find_variables(expression);
    std::unordered_set<sym::expr> direct_variables;
    for(auto runtime_source : context.runtime_extent_sources)
    {
        const auto& runtime_shape = runtime_source->get_shape();
        if(not runtime_shape.symbolic())
            continue;
        for(const auto& dimension : runtime_shape.dyn_dims())
            if(dimension.sym_expr.name() == "variable")
                direct_variables.insert(sym::as_symbol(dimension.sym_expr));
    }

    std::unordered_map<sym::expr, sym::expr> substitutions;
    for(const auto& variable : variables)
    {
        if(contains(direct_variables, variable))
            continue;
        auto value = context.values.find(variable);
        if(value == context.values.end())
            MIGRAPHX_THROW("SPLIT_SYM_DIM: runtime extent has no source for " +
                           variable.to_string());
        substitutions.emplace(variable, sym::lit(value->second));
    }
    auto runtime_expression = expression.subs(substitutions);
    auto runtime_variables  = sym::find_variables(runtime_expression);

    std::unordered_set<sym::expr> resolved_variables;
    std::vector<instruction_ref> expression_sources;
    std::copy_if(context.runtime_extent_sources.begin(),
                 context.runtime_extent_sources.end(),
                 std::back_inserter(expression_sources),
                 [&](instruction_ref runtime_source) {
                     const auto& runtime_shape = runtime_source->get_shape();
                     if(not runtime_shape.symbolic())
                         return false;
                     bool required = false;
                     for(const auto& dimension : runtime_shape.dyn_dims())
                     {
                         if(dimension.sym_expr.name() != "variable")
                             continue;
                         auto variable = sym::as_symbol(dimension.sym_expr);
                         if(contains(runtime_variables, variable))
                         {
                             resolved_variables.insert(variable);
                             required = true;
                         }
                     }
                     return required;
                 });
    if(resolved_variables.size() != runtime_variables.size())
        MIGRAPHX_THROW("SPLIT_SYM_DIM: runtime extent has unresolved runtime variables");

    if(not expression_sources.empty())
        return m.add_instruction(
            make_op("eval_expr_from_shape",
                    {{"expressions", to_value(std::vector<sym::expr>{runtime_expression})}}),
            expression_sources);

    const int64_t value = runtime_expression.eval_uint({});
    return m.add_literal(literal{shape{shape::int64_type, {1}}, std::vector<int64_t>{value}});
}

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
        {
            result.freezer = freeze;
            result.runtime_extent_expressions.push_back(data_dims.at(*axis).sym_expr);
        }
        return result;
    }

    static instruction_ref freeze(module& m,
                                  instruction_ref source,
                                  const std::vector<instruction_ref>& args,
                                  const freezer_context& context)
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

        const auto data     = args.front();
        const auto indices  = args.back();
        auto runtime_extent = add_runtime_extent(m, extent.sym_expr, context);
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
        auto result     = m.add_instruction(source->get_operator(), data, normalized);
        const auto& result_shape = result->get_shape();
        if(result_shape.symbolic() and result_shape.is_fixed() and result_shape.has_fixed_strides())
            instruction::replace(
                result, result->get_operator(), result_shape.to_static(), result->inputs());
        return result;
    }
};

} // namespace
} // namespace split_sym
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
