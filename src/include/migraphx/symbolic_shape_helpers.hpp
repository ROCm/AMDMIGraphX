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
#ifndef MIGRAPHX_GUARD_SYMBOLIC_SHAPE_HELPERS_HPP
#define MIGRAPHX_GUARD_SYMBOLIC_SHAPE_HELPERS_HPP

#include <migraphx/config.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/sym.hpp>

#include <optional>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;
struct shape;

using symbol_source_map = std::unordered_map<sym::expr, instruction_ref>;

// Returns the first expression that cannot be evaluated from the symbolic input dimensions.
MIGRAPHX_EXPORT std::optional<sym::expr>
find_unresolved_shape_expression(const std::vector<shape>& input_shapes,
                                 const std::vector<sym::expr>& expressions);

// Associates each directly exposed symbolic variable with its first source.
MIGRAPHX_EXPORT symbol_source_map
make_symbol_source_map(const std::vector<instruction_ref>& sources);

// Resolves expressions from `source_map`, preserving `source_order` in the result.
MIGRAPHX_EXPORT std::optional<std::vector<instruction_ref>>
find_expression_sources(const std::vector<sym::expr>& expressions,
                        const symbol_source_map& source_map,
                        const std::vector<instruction_ref>& source_order);

// Uses the ordered inputs directly when their dimensions resolve every expression. Otherwise
// searches the module for instructions that expose every free variable as a whole dimension.
MIGRAPHX_EXPORT std::optional<std::vector<instruction_ref>>
find_expression_sources(const module& m,
                        const std::vector<instruction_ref>& inputs,
                        const std::vector<sym::expr>& expressions);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
