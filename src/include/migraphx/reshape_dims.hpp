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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_RESHAPE_DIMS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_RESHAPE_DIMS_HPP

#include <migraphx/config.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/dim_like.hpp>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct reshape_dims_options
{
    bool lazy = false;
};

// nullopt when the layout can't be proven; the caller falls back to standard.
MIGRAPHX_EXPORT optional<shape>
reshape_dims(const shape& input, const std::vector<sym::expr>& rdims, reshape_dims_options options);

// Resolve reshape `dims` entries against a symbolic input: 0 copies the input dim,
// -1 is inferred as the leftover element count, literals/symbols are taken as-is.
MIGRAPHX_EXPORT std::vector<shape::dynamic_dimension>
resolve_reshape_dims(const shape& sym_in, const std::vector<dim_like>& dims);

// Throws if `dims` holds a range-based dynamic_dimension or more than one -1 entry.
MIGRAPHX_EXPORT void validate_reshape_dims(const std::string& name,
                                           const std::vector<dim_like>& dims);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_RESHAPE_DIMS_HPP
