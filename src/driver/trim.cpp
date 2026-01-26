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
 *
 */
#include "trim.hpp"
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <unordered_map>
#include <unordered_set>

namespace migraphx {
namespace driver {
inline namespace MIGRAPHX_INLINE_NS {

static instruction_ref capture_arg(std::unordered_set<instruction_ref>& s, instruction_ref ins)
{
    auto aliases = instruction::get_output_alias(ins, true);
    if(aliases.size() == 1 and aliases.front() != ins)
    {
        s.insert(ins);
        return capture_arg(s, aliases.front());
    }
    if(contains({"reshape", "contiguous"}, ins->name()))
    {
        s.insert(ins);
        return capture_arg(s, ins->inputs().front());
    }
    return ins;
}

static instruction_ref add_placeholder(module& m, instruction_ref ins)
{
    if(ins->inputs().empty())
        return ins;
    if(ins->can_eval())
    {
        auto e = ins->eval();
        return m.add_literal(literal{e.get_shape(), e.data()});
    }
    return m.add_parameter("x" + std::to_string(m.get_parameters().size()), ins->get_shape());
}

void trim_module(module& m, std::size_t loc, std::size_t n)
{
    if(loc > m.size())
        MIGRAPHX_THROW("Trim out of range.");
    auto last  = std::prev(m.end(), loc);
    auto start = std::prev(last, n);
    m.remove_instructions(last, m.end());
    if(n == 0)
        return;
    if(n > m.size())
        MIGRAPHX_THROW("Trim size out of range.");
    std::unordered_map<instruction_ref, instruction_ref> map_ins;
    std::unordered_set<instruction_ref> instruction_set;
    auto instructions = range(start, m.end());
    for(instruction_ref ins : iterator_for(instructions))
    {
        instruction_set.insert(ins);
        for(auto input : ins->inputs())
        {
            if(contains(instruction_set, input))
                continue;
            auto arg         = capture_arg(instruction_set, input);
            auto placeholder = add_placeholder(m, arg);
            assert(placeholder->get_shape() == arg->get_shape());
            if(placeholder == arg)
                continue;
            instruction_set.insert(placeholder);
            map_ins[arg] = placeholder;
        }
    }
    for(auto [old_ins, new_ins] : map_ins)
        m.replace_instruction(old_ins, new_ins);
    run_passes(m, {dead_code_elimination{}});
    for(auto pins : m.get_parameters())
    {
        if(not pins->outputs().empty())
            continue;
        m.remove_instruction(pins);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace driver
} // namespace migraphx
