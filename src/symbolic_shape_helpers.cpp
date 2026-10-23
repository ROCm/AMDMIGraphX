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

#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

using substitutions = std::unordered_map<sym::expr, sym::expr>;

void add_shape_expressions(substitutions& available, instruction_ref source)
{
    const auto& s = source->get_shape();
    if(not s.symbolic())
        return;
    std::vector<std::pair<sym::expr, sym::expr>> expressions;
    transform_if(
        s.dyn_dims().begin(),
        s.dyn_dims().end(),
        std::back_inserter(expressions),
        [](const auto& d) { return d.is_symbolic(); },
        [](const auto& d) { return std::make_pair(d.sym_expr, sym::lit(1)); });
    transform_if(
        s.dyn_dims().begin(),
        s.dyn_dims().end(),
        std::back_inserter(expressions),
        [](const auto& d) { return d.is_symbolic() and d.sym_expr.name() == "variable"; },
        [](const auto& d) { return std::make_pair(sym::as_symbol(d.sym_expr), sym::lit(1)); });
    available.insert(expressions.begin(), expressions.end());
}

std::size_t count_unresolved(const std::vector<sym::expr>& expressions,
                             const substitutions& available)
{
    return std::accumulate(expressions.begin(),
                           expressions.end(),
                           std::unordered_set<sym::expr>{},
                           [&](auto unresolved, const auto& expression) {
                               auto variables = sym::find_variables(expression.subs(available));
                               unresolved.insert(variables.begin(), variables.end());
                               return unresolved;
                           })
        .size();
}

struct source_candidate
{
    instruction_ref source;
    std::size_t unresolved;
    substitutions available;
};

} // namespace

std::optional<std::vector<instruction_ref>> find_expression_sources(
    const module& m, instruction_ref data, const std::vector<sym::expr>& expressions)
{
    std::vector<instruction_ref> candidates = {data};
    auto parameters                         = m.get_parameters();
    candidates.insert(candidates.end(), parameters.begin(), parameters.end());
    auto instructions = iterator_for(m);
    candidates.insert(candidates.end(), instructions.begin(), instructions.end());

    std::vector<instruction_ref> result = {data};
    substitutions available;
    add_shape_expressions(available, data);
    auto unresolved = count_unresolved(expressions, available);

    while(unresolved > 0)
    {
        auto best = std::accumulate(
            candidates.begin(),
            candidates.end(),
            std::optional<source_candidate>{},
            [&](auto current, auto candidate) {
                if(contains(result, candidate))
                    return current;
                auto candidate_available = available;
                add_shape_expressions(candidate_available, candidate);
                auto candidate_unresolved = count_unresolved(expressions, candidate_available);
                auto current_unresolved   = current.has_value() ? current->unresolved : unresolved;
                if(candidate_unresolved >= current_unresolved)
                    return current;
                return std::optional<source_candidate>{
                    {candidate, candidate_unresolved, std::move(candidate_available)}};
            });
        if(not best.has_value())
            return std::nullopt;
        result.push_back(best->source);
        available  = std::move(best->available);
        unresolved = best->unresolved;
    }
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
