/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_INSTRUCTION_TRAVERSAL_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_INSTRUCTION_TRAVERSAL_HPP

#include <migraphx/config.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/unfold.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

inline auto get_output_path(instruction_ref ins)
{
    return unfold(ins, [](instruction_ref out) -> std::optional<instruction_ref> {
        if(out->outputs().size() != 1)
            return std::nullopt;
        return out->outputs().front();
    });
}

inline auto get_input_path(instruction_ref ins)
{
    return unfold(ins, [](instruction_ref in) -> std::optional<instruction_ref> {
        if(in->inputs().size() != 1)
            return std::nullopt;
        return in->inputs().front();
    });
}

// The instructions that share the buffer of `ins`, starting with `ins` and ending with the
// instruction that owns the buffer, such as an allocation or a parameter. The path stops early
// when an instruction aliases more than one input since there is no single buffer to follow.
inline auto get_alias_path(instruction_ref ins)
{
    return unfold(ins, [](instruction_ref in) -> std::optional<instruction_ref> {
        auto aliases = instruction::get_output_alias(in, true);
        if(aliases.size() != 1 or aliases.front() == in)
            return std::nullopt;
        // An instruction_ref points into the module and not into the local vector
        // cppcheck-suppress returnDanglingLifetime
        return aliases.front();
    });
}

// A lowered select_module returns a tuple backed by one trailing buffer per tuple element.
// Map get_tuple_elem directly to its corresponding buffer because generic alias traversal
// returns every buffer backing the tuple.
inline optional<instruction_ref> get_select_module_buffer(instruction_ref ins)
{
    auto tuple_op         = ins->get_operator();
    auto tuple_attributes = tuple_op.to_value();
    if(tuple_op.name() != "get_tuple_elem")
    {
        if(not tuple_attributes.contains("name") or not tuple_attributes.contains("operator"))
            return nullopt;
        const auto& wrapped_name = tuple_attributes.at("name");
        if(not wrapped_name.is_string() or wrapped_name.get_string() != "get_tuple_elem")
            return nullopt;
        tuple_attributes = tuple_attributes.at("operator");
    }
    auto producer = ins->inputs().front();
    if(producer->name() != "select_module")
        return nullopt;
    const auto& inputs = producer->inputs();
    auto aliases       = producer->get_operator().output_alias(to_shapes(inputs));
    auto index         = tuple_attributes.at("index").to<std::size_t>();
    if(index >= aliases.size())
        return nullopt;
    return inputs.at(aliases.at(index));
}

inline std::vector<instruction_ref> resolve_output_aliases(instruction_ref ins)
{
    if(auto buffer = get_select_module_buffer(ins))
        return resolve_output_aliases(*buffer);
    auto aliases = instruction::get_output_alias(ins, true);
    if(aliases.size() == 1 and aliases.front() == ins)
        return {ins};
    std::vector<instruction_ref> result;
    for(auto alias : aliases)
    {
        auto resolved = resolve_output_aliases(alias);
        result.insert(result.end(), resolved.begin(), resolved.end());
    }
    return result;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_INSTRUCTION_TRAVERSAL_HPP
