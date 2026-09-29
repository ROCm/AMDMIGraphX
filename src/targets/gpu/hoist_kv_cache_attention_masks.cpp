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
#include <migraphx/gpu/hoist_kv_cache_attention_masks.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

namespace {

bool is_index_type(const shape& s)
{
    return contains({shape::int32_type, shape::int64_type}, s.type());
}

bool is_pointwise(instruction_ref ins)
{
    return ins->get_operator().attributes().get("pointwise", false);
}

bool is_kv_cache_attention(instruction_ref ins)
{
    return ins->name() == "group" and
           ins->get_operator().to_value().at("tag").to<std::string>() == "kv_cache_attention";
}

// The integer pointwise instructions of the module that are computed only from its parameters,
// its constants, and other integer instructions, along with the converts of their results.
// Constants are not included: the parent gets its own copy of the ones it needs.
std::unordered_set<instruction_ref> find_index_arithmetic(const module& m)
{
    std::unordered_set<instruction_ref> computable;
    std::unordered_set<instruction_ref> result;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "@param" or ins->can_eval())
        {
            computable.insert(ins);
            continue;
        }
        if(not all_of(ins->inputs(),
                      [&](instruction_ref input) { return contains(computable, input); }))
            continue;
        const bool hoist = is_pointwise(ins) and
                           (is_index_type(ins->get_shape()) or
                            (ins->name() == "convert" and contains(result, ins->inputs().front())));
        if(hoist)
            result.insert(ins);
        if(hoist or is_index_type(ins->get_shape()))
            computable.insert(ins);
    }
    return result;
}

// Instructions reachable from roots through inputs, stopping at (and excluding) the ones that
// stop returns true for, in module order.
template <class Stop>
std::vector<instruction_ref>
reachable_inputs(const module& m, const std::vector<instruction_ref>& roots, Stop stop)
{
    std::unordered_set<instruction_ref> visited;
    fix([&](auto self, const std::vector<instruction_ref>& inss) {
        for(auto ins : inss)
        {
            if(stop(ins) or not visited.insert(ins).second)
                continue;
            self(ins->inputs());
        }
    })(roots);
    std::vector<instruction_ref> result;
    copy_if(iterator_for(m), std::back_inserter(result), [&](instruction_ref ins) {
        return contains(visited, ins);
    });
    return result;
}

void hoist_index_arithmetic(module& m, instruction_ref group)
{
    module_ref attn = group->module_inputs().front();
    auto hoisted    = find_index_arithmetic(*attn);
    if(hoisted.empty())
        return;
    auto is_param = [](instruction_ref ins) { return ins->name() == "@param"; };

    auto returns = std::prev(attn->end())->inputs();
    auto kept    = reachable_inputs(*attn, returns, [&](instruction_ref ins) {
        return is_param(ins) or contains(hoisted, ins);
    });
    std::vector<instruction_ref> boundaries;
    copy_if(hoisted, std::back_inserter(boundaries), [&](instruction_ref ins) {
        return any_of(kept, [&](instruction_ref k) { return contains(k->inputs(), ins); });
    });

    auto map_ins = attn->get_ins_param_map(group->inputs(), true);
    m.insert_instructions(group, reachable_inputs(*attn, boundaries, is_param), &map_ins);

    module new_attn{attn->name()};
    std::unordered_map<instruction_ref, instruction_ref> map_new;
    new_attn.fuse(kept, &map_new);
    std::vector<instruction_ref> new_returns;
    std::transform(returns.begin(),
                   returns.end(),
                   std::back_inserter(new_returns),
                   [&](instruction_ref ins) { return map_new.at(ins); });
    new_attn.add_return(new_returns);

    std::unordered_map<instruction_ref, instruction_ref> new_to_parent;
    transform_if(
        map_new.begin(),
        map_new.end(),
        std::inserter(new_to_parent, new_to_parent.end()),
        [&](const auto& p) { return is_param(p.second) and contains(map_ins, p.first); },
        [&](const auto& p) { return std::make_pair(p.second, map_ins.at(p.first)); });
    auto new_inputs = new_attn.get_inputs(new_to_parent);

    new_attn.set_bypass(attn->bypass());
    *attn = std::move(new_attn);
    m.replace_instruction(group, group->get_operator(), new_inputs, {attn});
}

} // namespace

void hoist_kv_cache_attention_masks::apply(module& m) const
{
    std::vector<instruction_ref> groups;
    copy_if(iterator_for(m), std::back_inserter(groups), &is_kv_cache_attention);
    for(auto group : groups)
        hoist_index_arithmetic(m, group);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
