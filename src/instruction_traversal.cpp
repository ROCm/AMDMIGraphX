/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/instruction_traversal.hpp>
#include <migraphx/output_iterator.hpp>
#include <migraphx/value.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static optional<instruction_ref> get_select_module_buffer(instruction_ref ins)
{
    if(ins->inputs().empty() or ins->inputs().front()->name() != "select_module")
        return nullopt;
    auto tuple_attributes = ins->get_operator().to_value();
    if(ins->name() != "get_tuple_elem")
    {
        // A lowered wrapper such as ref::op serializes the wrapped operator under these keys.
        if(not tuple_attributes.contains("name") or not tuple_attributes.contains("operator"))
            return nullopt;
        const auto& wrapped_name = tuple_attributes.at("name");
        if(not wrapped_name.is_string() or wrapped_name.get_string() != "get_tuple_elem")
            return nullopt;
        tuple_attributes = tuple_attributes.at("operator");
    }
    auto producer      = ins->inputs().front();
    const auto& inputs = producer->inputs();
    auto aliases       = producer->get_operator().output_alias(to_shapes(inputs));
    auto index         = tuple_attributes.at("index").to<std::size_t>();
    if(index >= aliases.size())
        return nullopt;
    return inputs.at(aliases.at(index));
}

std::vector<instruction_ref> resolve_output_aliases(instruction_ref ins)
{
    if(auto buffer = get_select_module_buffer(ins))
        return resolve_output_aliases(*buffer);
    auto aliases = instruction::get_output_alias(ins, true);
    if(aliases.size() == 1 and aliases.front() == ins)
        return {ins};
    std::vector<instruction_ref> result;
    std::transform(aliases.begin(), aliases.end(), join_back_inserter(result), [](auto alias) {
        return resolve_output_aliases(alias);
    });
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
