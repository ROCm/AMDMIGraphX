/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/promote_literals.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static std::vector<instruction_ref> find_literals(module& m)
{
    std::vector<instruction_ref> result;
    auto instructions = iterator_for(m);
    std::copy_if(instructions.begin(),
                 instructions.end(),
                 std::back_inserter(result),
                 [](instruction_ref ins) { return ins->name() == "@literal"; });
    return result;
}

static void add_submodules(module_ref m, std::unordered_set<module_ref>& result)
{
    if(not result.insert(m).second)
        return;
    for(auto ins : iterator_for(*m))
        for(auto* sub : ins->module_inputs())
            add_submodules(sub, result);
}

/**
 * Collects the modules below a `select_module`. Those are specializations of one graph, so they
 * are the modules expected to hold equal copies of the same value.
 */
static std::unordered_set<module_ref> find_specializations(module& root)
{
    std::unordered_set<module_ref> result;
    for(auto ins : iterator_for(root))
        if(ins->name() == "select_module")
            for(auto* sub : ins->module_inputs())
                add_submodules(sub, result);
    return result;
}

void promote_literals::apply(module_pass_manager& mpm) const
{
    module& m              = mpm.get_module();
    module_ref root_module = mpm.get_root_module();
    if(m == *root_module)
        return;

    // Specializations are optimized independently, so each one can fold its own copy of a value
    // they all share. Reusing an equal literal already promoted to the root module keeps one copy
    // instead of adding a duplicate per specialization. This is limited to specializations
    // because only they are known to duplicate a value that used to be shared; elsewhere two
    // equal literals are left alone. Comparing literals checks the shape before the data, so
    // unequal ones are rejected without reading their buffers.
    const bool share = contains(find_specializations(*root_module), &m);
    auto promoted    = share ? find_literals(*root_module) : std::vector<instruction_ref>{};

    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "@literal")
        {
            auto it = std::find_if(promoted.begin(), promoted.end(), [&](instruction_ref lit) {
                return lit->get_literal() == ins->get_literal();
            });
            instruction_ref new_lit{};
            if(it == promoted.end())
            {
                new_lit = root_module->add_literal(ins->get_literal());
                if(share)
                    promoted.push_back(new_lit);
            }
            else
            {
                new_lit = *it;
            }
            auto ins_outputs = ins->outputs();
            for(auto out_ins : ins_outputs)
            {
                migraphx::instruction::replace_argument(out_ins, ins, new_lit);
            }
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
