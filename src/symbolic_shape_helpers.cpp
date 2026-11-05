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
#include <migraphx/symbolic_shape_helpers.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape.hpp>

#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

template <class Range>
void append_unique_sources(std::vector<instruction_ref>& result,
                           std::unordered_set<instruction_ref>& seen,
                           const Range& sources)
{
    std::copy_if(sources.begin(), sources.end(), std::back_inserter(result), [&](auto source) {
        return seen.insert(source).second;
    });
}

std::vector<std::pair<std::string, sym::expr>>
find_ordered_variables(const std::vector<sym::expr>& expressions)
{
    std::unordered_set<sym::expr> required;
    for(const auto& expression : expressions)
    {
        auto variables = sym::find_variables(expression);
        required.insert(variables.begin(), variables.end());
    }

    std::vector<std::pair<std::string, sym::expr>> result;
    std::transform(
        required.begin(), required.end(), std::back_inserter(result), [](const auto& variable) {
            return std::make_pair(variable.to_string(), variable);
        });
    // Keep eval_expr_from_shape inputs deterministic despite the unordered variable set.
    std::sort(result.begin(), result.end(), [](const auto& x, const auto& y) {
        return x.first < y.first;
    });
    return result;
}

std::optional<std::vector<instruction_ref>> find_expression_sources_by_variable(
    const std::vector<std::pair<std::string, sym::expr>>& ordered_variables,
    const symbol_source_map& source_map)
{
    std::vector<instruction_ref> result;
    std::unordered_set<instruction_ref> seen;
    for(const auto& ordered_variable : ordered_variables)
    {
        const auto& variable = ordered_variable.second;
        auto source          = source_map.find(variable);
        if(source == source_map.end())
            return std::nullopt;
        if(seen.insert(source->second).second)
            result.push_back(source->second);
    }
    return result;
}

} // namespace

std::optional<sym::expr> find_unresolved_shape_expression(const std::vector<shape>& input_shapes,
                                                          const std::vector<sym::expr>& expressions)
{
    // Substitution only needs to remove expressions supplied by input dimensions.
    const auto available_value = sym::lit(1);
    std::unordered_map<sym::expr, sym::expr> available;
    for(const auto& input_shape : input_shapes)
    {
        if(not input_shape.symbolic())
            continue;
        std::transform(input_shape.dyn_dims().begin(),
                       input_shape.dyn_dims().end(),
                       std::inserter(available, available.end()),
                       [&](const auto& d) {
                           return std::make_pair(sym::as_symbol(d.sym_expr), available_value);
                       });
    }

    auto unresolved = std::find_if(expressions.begin(), expressions.end(), [&](const auto& e) {
        return not sym::find_variables(e.subs(available)).empty();
    });
    if(unresolved == expressions.end())
        return std::nullopt;
    return *unresolved;
}

symbol_source_map make_symbol_source_map(const std::vector<instruction_ref>& sources)
{
    symbol_source_map result;
    for(auto source : sources)
    {
        const auto& s = source->get_shape();
        if(not s.symbolic())
            continue;
        for(const auto& d : s.dyn_dims())
            if(d.sym_expr.name() == "variable")
                result.emplace(sym::as_symbol(d.sym_expr), source);
    }
    return result;
}

std::optional<std::vector<instruction_ref>>
find_expression_sources(const std::vector<sym::expr>& expressions,
                        const symbol_source_map& source_map,
                        const std::vector<instruction_ref>& source_order)
{
    auto sources =
        find_expression_sources_by_variable(find_ordered_variables(expressions), source_map);
    if(not sources.has_value())
        return std::nullopt;

    std::unordered_set<instruction_ref> required{sources->begin(), sources->end()};
    std::vector<instruction_ref> result;
    std::copy_if(source_order.begin(),
                 source_order.end(),
                 std::back_inserter(result),
                 [&](auto source) { return contains(required, source); });
    return result;
}

std::optional<std::vector<instruction_ref>>
find_expression_sources(const module& m,
                        const std::vector<instruction_ref>& inputs,
                        const std::vector<sym::expr>& expressions)
{
    std::vector<instruction_ref> unique_inputs;
    std::unordered_set<instruction_ref> seen;
    append_unique_sources(unique_inputs, seen, inputs);

    auto ordered_variables = find_ordered_variables(expressions);
    if(ordered_variables.empty() and not unique_inputs.empty())
        return std::vector<instruction_ref>{unique_inputs.front()};

    std::vector<shape> input_shapes;
    std::transform(unique_inputs.begin(),
                   unique_inputs.end(),
                   std::back_inserter(input_shapes),
                   [](auto input) { return input->get_shape(); });
    if(not unique_inputs.empty() and
       not find_unresolved_shape_expression(input_shapes, expressions).has_value())
        return unique_inputs;

    std::vector<instruction_ref> candidates = unique_inputs;
    auto parameters                         = m.get_parameters();
    append_unique_sources(candidates, seen, parameters);
    append_unique_sources(candidates, seen, iterator_for(m));
    if(ordered_variables.empty())
    {
        if(candidates.empty())
            return std::nullopt;
        return std::vector<instruction_ref>{candidates.front()};
    }

    return find_expression_sources_by_variable(ordered_variables,
                                               make_symbol_source_map(candidates));
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
