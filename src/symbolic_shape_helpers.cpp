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

#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

// Check if all expressions are resolved by the inputs.
bool shapes_resolve_expressions(const std::vector<instruction_ref>& inputs,
                                const std::vector<sym::expr>& expressions)
{
    // sym::lit(1) is a dummy value for sym_expr::subs()
    std::unordered_map<sym::expr, sym::expr> available;
    for(auto input : inputs)
    {
        const auto& input_shape = input->get_shape();
        if(not input_shape.symbolic())
            continue;
        transform_if(
            input_shape.dyn_dims().begin(),
            input_shape.dyn_dims().end(),
            std::inserter(available, available.end()),
            [](const auto& d) { return d.is_symbolic(); },
            [](const auto& d) { return std::make_pair(sym::as_symbol(d.sym_expr), sym::lit(1)); });
    }
    return all_of(expressions, [&](const auto& expression) {
        return sym::find_variables(expression.subs(available)).empty();
    });
}

bool directly_contains_symbol(instruction_ref source, const sym::expr& symbol)
{
    const auto& s = source->get_shape();
    if(not s.symbolic())
        return false;
    return any_of(s.dyn_dims(), [&](const auto& d) {
        return d.is_symbolic() and d.sym_expr.name() == "variable" and
               sym::same_symbol(d.sym_expr, symbol);
    });
}

} // namespace

std::optional<std::vector<instruction_ref>>
find_expression_sources(const module& m,
                        const std::vector<instruction_ref>& inputs,
                        const std::vector<sym::expr>& expressions)
{
    std::vector<instruction_ref> unique_inputs;
    std::copy_if(inputs.begin(), inputs.end(), std::back_inserter(unique_inputs), [&](auto input) {
        return not contains(unique_inputs, input);
    });

    std::unordered_set<sym::expr> required;
    for(const auto& expression : expressions)
    {
        auto variables = sym::find_variables(expression);
        required.insert(variables.begin(), variables.end());
    }
    if(required.empty() and not unique_inputs.empty())
        return std::vector<instruction_ref>{unique_inputs.front()};
    if(not unique_inputs.empty() and shapes_resolve_expressions(unique_inputs, expressions))
        return unique_inputs;

    std::vector<std::pair<std::string, sym::expr>> ordered_variables;
    std::transform(
        required.begin(),
        required.end(),
        std::back_inserter(ordered_variables),
        [](const auto& variable) { return std::make_pair(variable.to_string(), variable); });
    std::sort(ordered_variables.begin(), ordered_variables.end(), [](const auto& x, const auto& y) {
        return x.first < y.first;
    });

    std::vector<instruction_ref> candidates = unique_inputs;
    auto parameters                         = m.get_parameters();
    std::copy_if(parameters.begin(),
                 parameters.end(),
                 std::back_inserter(candidates),
                 [&](auto parameter) { return not contains(candidates, parameter); });
    auto instructions = iterator_for(m);
    std::copy_if(instructions.begin(),
                 instructions.end(),
                 std::back_inserter(candidates),
                 [&](auto instruction) { return not contains(candidates, instruction); });
    if(required.empty())
    {
        if(candidates.empty())
            return std::nullopt;
        return std::vector<instruction_ref>{candidates.front()};
    }

    std::vector<instruction_ref> result;
    for(const auto& ordered_variable : ordered_variables)
    {
        const auto& variable = ordered_variable.second;
        auto source = std::find_if(candidates.begin(), candidates.end(), [&](auto candidate) {
            return directly_contains_symbol(candidate, variable);
        });
        if(source == candidates.end())
            return std::nullopt;
        if(not contains(result, *source))
            result.push_back(*source);
    }
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
