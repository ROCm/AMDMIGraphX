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
#include <migraphx/gpu/limit_concat.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/serialize.hpp>
#include <algorithm>
#include <cassert>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace {

std::size_t tensor_count(instruction_ref ins, std::size_t additional)
{
    return ins->inputs().size() - additional;
}

// One group of the wide concat. A leftover single input is not a concat; the
// caller passes it through and joins it at the next level.
instruction_ref make_chunk(module& m,
                           instruction_ref pos,
                           const operation& concat_op,
                           const std::vector<instruction_ref>& group)
{
    std::vector<shape> shapes;
    std::transform(group.begin(), group.end(), std::back_inserter(shapes), [](instruction_ref in) {
        return in->get_shape();
    });
    auto out   = concat_op.compute_shape(shapes);
    auto alloc = m.insert_instruction(pos, make_op("allocate", {{"shape", to_value(out)}}));
    auto args  = group;
    args.push_back(alloc);
    return m.insert_instruction(
        pos, make_op("gpu::precompile_op", {{"op", to_value(concat_op)}}), args);
}

// Reduce `cur` until it fits in one kernel. Each step replaces a run of inputs
// with one concat, so the count strictly decreases while max_inputs >= 2.
std::vector<instruction_ref> pack_to_limit(module& m,
                                           instruction_ref pos,
                                           const operation& concat_op,
                                           std::vector<instruction_ref> cur,
                                           std::size_t max_inputs)
{
    while(cur.size() > max_inputs)
    {
        std::vector<instruction_ref> next;
        std::size_t i = 0;
        while(i < cur.size())
        {
            auto take = std::min(max_inputs, cur.size() - i);
            if(take < 2)
            {
                next.push_back(cur[i]);
                break;
            }
            next.push_back(
                make_chunk(m, pos, concat_op, {cur.begin() + i, cur.begin() + i + take}));
            i += take;
        }
        cur = std::move(next);
    }
    return cur;
}

void split_concat(module& m, instruction_ref ins, std::size_t max_inputs)
{
    auto v          = ins->get_operator().to_value();
    auto concat_op  = from_value<operation>(v.at("op"));
    auto additional = v.at("additional_args").to<std::size_t>();
    if(concat_op.name() != "concat" or ins->inputs().size() < additional + 2)
        return;
    if(tensor_count(ins, additional) <= max_inputs)
        return;

    const auto& inputs = ins->inputs();
    std::vector<instruction_ref> tensors(inputs.begin(), inputs.end() - additional);
    std::vector<instruction_ref> trailing(inputs.end() - additional, inputs.end());
    auto packed = pack_to_limit(m, ins, concat_op, std::move(tensors), max_inputs);
    assert(packed.size() >= 2);

    auto args = std::move(packed);
    args.insert(args.end(), trailing.begin(), trailing.end());
    m.replace_instruction(ins, ins->get_operator(), args, ins->module_inputs());
}

} // namespace

void limit_concat_inputs::apply(module& m) const
{
    // A concat of one input is not a concat, so the group size cannot be 1.
    const auto limit = std::max<std::size_t>(2, max_inputs);
    std::vector<instruction_ref> wide;
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "gpu::precompile_op")
            continue;
        auto v = ins->get_operator().to_value();
        if(not v.contains("op") or not v.contains("additional_args"))
            continue;
        if(v.at("op").at("name").to<std::string>() != "concat")
            continue;
        auto additional = v.at("additional_args").to<std::size_t>();
        if(ins->inputs().size() > additional + limit)
            wide.push_back(ins);
    }
    std::for_each(
        wide.begin(), wide.end(), [&](instruction_ref ins) { split_concat(m, ins, limit); });
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
