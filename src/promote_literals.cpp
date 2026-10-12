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

namespace {

// group literals by their shape to reduce the number of literal comparisons
struct literal_group
{
    shape s;
    std::vector<instruction_ref> literals;
};

using literal_groups = std::vector<literal_group>;

literal_group* find_literal_group(literal_groups& groups, const shape& s)
{
    auto group = std::find_if(
        groups.begin(), groups.end(), [&](const auto& candidate) { return candidate.s == s; });
    return group == groups.end() ? nullptr : &*group;
}

void append_root_literal(literal_groups& groups, instruction_ref ins)
{
    auto* group = find_literal_group(groups, ins->get_shape());
    if(group == nullptr)
        groups.push_back({ins->get_shape(), {ins}});
    else
        group->literals.push_back(ins);
}

void prepend_root_literal(literal_groups& groups, instruction_ref ins)
{
    auto* group = find_literal_group(groups, ins->get_shape());
    if(group == nullptr)
        groups.insert(groups.begin(), {ins->get_shape(), {ins}});
    else
        group->literals.insert(group->literals.begin(), ins);
}

literal_groups group_root_literals(module& root)
{
    literal_groups result;
    for(auto ins : iterator_for(root))
        if(ins->name() == "@literal")
            append_root_literal(result, ins);
    return result;
}

instruction_ref
find_or_promote_literal(module& root, literal_groups& groups, const literal& literal)
{
    auto* group = find_literal_group(groups, literal.get_shape());
    if(group != nullptr)
    {
        auto existing =
            std::find_if(group->literals.begin(), group->literals.end(), [&](auto root_literal) {
                return root_literal->get_literal() == literal;
            });
        if(existing != group->literals.end())
            return *existing;
    }

    auto result = root.add_literal(literal);
    prepend_root_literal(groups, result);
    return result;
}

void promote_module_literals(module& m, module& root, literal_groups& groups)
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "@literal")
            continue;

        auto new_literal = find_or_promote_literal(root, groups, ins->get_literal());
        auto outputs     = ins->outputs();
        for(auto output : outputs)
            instruction::replace_argument(output, ins, new_literal);
    }
}

} // namespace

void promote_literals::apply(module_pass_manager& mpm) const
{
    module& m              = mpm.get_module();
    module_ref root_module = mpm.get_root_module();
    if(&m != root_module)
        return;

    auto groups     = group_root_literals(*root_module);
    auto submodules = root_module->get_sub_modules();
    std::unordered_set<module_ref> visited;
    for(auto* submodule : reverse(submodules))
    {
        if(submodule->bypass() or not visited.insert(submodule).second)
            continue;
        promote_module_literals(*submodule, *root_module, groups);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
