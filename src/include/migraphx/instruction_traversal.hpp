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
#include <migraphx/unfold.hpp>
#include <algorithm>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

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

// Follows the input path of `ins` while `pred` holds. Returns the first instruction where it
// doesn't, along with the operators that were passed in the order they are applied. If `pred`
// holds for the whole path, returns `ins` with no operators.
template <class Predicate>
std::pair<instruction_ref, std::vector<operation>> get_input_ops_if(instruction_ref ins,
                                                                    Predicate pred)
{
    auto path = get_input_path(ins);
    auto it   = std::find_if_not(path.begin(), path.end(), pred);
    if(it == path.end())
        return {ins, {}};
    std::vector<operation> ops;
    std::transform(path.begin(), it, std::back_inserter(ops), [](instruction_ref x) {
        return x->get_operator();
    });
    std::reverse(ops.begin(), ops.end());
    return std::make_pair(*it, std::move(ops));
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

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_INSTRUCTION_TRAVERSAL_HPP
