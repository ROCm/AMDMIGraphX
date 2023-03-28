/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/propagate_constant.hpp>
#include <migraphx/program.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/par_for.hpp>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool skip_propogate(instruction_ref ins)
{
    if(ins->name() == "contiguous")
        return skip_propogate(ins->inputs().front());
    auto&& s = ins->get_shape();
    if(s.broadcasted() and not s.scalar())
        return true;
    if(s.scalar() and s.elements() != 1)
        return true;
    return false;
}

bool is_const(instruction_ref ins) { return ins->can_eval() and not skip_propogate(ins); }

void propagate_constant::apply(module& m) const
{
    std::unordered_set<instruction_ref> const_instrs;
    auto last = std::prev(m.end());

    // Find instructions that can be evaluated to a literal
    for(auto i : iterator_for(m))
    {
        if(is_const(i) and i != last)
            continue;

        std::copy_if(
            i->inputs().begin(),
            i->inputs().end(),
            std::inserter(const_instrs, const_instrs.begin()),
            [&](const instruction_ref ins) { return is_const(ins) and ins->name() != "@literal"; });
    }

    // Compute literals in parallel
    std::vector<instruction_ref> const_instrs_vec{const_instrs.begin(), const_instrs.end()};
    std::vector<argument> literals(const_instrs_vec.size());
    par_for(const_instrs_vec.size(), 1, [&](const auto i) {
        literals[i] = const_instrs_vec[i]->eval();
    });

    // Replace instructions in m
    for(size_t i = 0; i < const_instrs_vec.size(); i++)
    {
        if(not literals[i].empty())
        {
            assert(literals[i].get_shape() == const_instrs_vec[i]->get_shape());
            auto l = m.add_literal(literals[i].get_shape(), literals[i].data());
            m.replace_instruction(const_instrs_vec[i], l);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
