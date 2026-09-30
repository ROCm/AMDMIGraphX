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
#include <migraphx/onnx/symbolic_shape.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>

#include <algorithm>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

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

std::optional<std::vector<instruction_ref>> find_expression_sources(
    const module& m, instruction_ref data, const std::vector<sym::expr>& expressions)
{
    std::unordered_set<sym::expr> required;
    for(const auto& expression : expressions)
    {
        auto variables = sym::find_variables(expression);
        required.insert(variables.begin(), variables.end());
    }

    std::vector<std::pair<std::string, sym::expr>> ordered_variables;
    std::transform(
        required.begin(),
        required.end(),
        std::back_inserter(ordered_variables),
        [](const auto& variable) { return std::make_pair(variable.to_string(), variable); });
    std::sort(ordered_variables.begin(), ordered_variables.end(), [](const auto& x, const auto& y) {
        return x.first < y.first;
    });

    std::vector<instruction_ref> candidates = {data};
    auto parameters                         = m.get_parameters();
    candidates.insert(candidates.end(), parameters.begin(), parameters.end());
    auto instructions = iterator_for(m);
    candidates.insert(candidates.end(), instructions.begin(), instructions.end());

    std::vector<instruction_ref> result;
    for(const auto& [name, variable] : ordered_variables)
    {
        (void)name;
        auto source = std::find_if(candidates.begin(), candidates.end(), [&](auto candidate) {
            return directly_contains_symbol(candidate, variable);
        });
        if(source == candidates.end())
            return std::nullopt;
        if(not contains(result, *source))
            result.push_back(*source);
    }
    if(result.empty())
        result.push_back(data);
    return result;
}

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
