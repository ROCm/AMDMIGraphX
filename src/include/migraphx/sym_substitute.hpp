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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_SYM_SUBSTITUTE_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SYM_SUBSTITUTE_HPP

#include <migraphx/config.hpp>
#include <migraphx/export.h>
#include <migraphx/optional.hpp>
#include <migraphx/dim_like.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/sym.hpp>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// The size each symbol takes in one specialization of a symbolic graph.
using symbol_map = std::unordered_map<sym::expr, std::size_t>;

namespace detail {

using expr_map = std::unordered_map<sym::expr, sym::expr>;

MIGRAPHX_EXPORT expr_map as_expr_map(const symbol_map& symbols);

// Substitution is total: a symbol missing from the map is left alone rather than treated as an
// error, so a partial map specializes what it can and the rest stays symbolic.
MIGRAPHX_EXPORT void substitute(sym::expr& e, const expr_map& symbols);
MIGRAPHX_EXPORT void substitute(shape::dynamic_dimension& dd, const expr_map& symbols);
MIGRAPHX_EXPORT void substitute(shape& s, const expr_map& symbols);
// A dim entry substitutes like the dimension it holds, and collapses to a plain size once that
// resolves, so an operator targeting symbolic dims becomes static along with its input.
MIGRAPHX_EXPORT void substitute(dim_like& d, const expr_map& symbols);

// Anything that cannot hold a symbol is left as is, which is what makes the reflection walk
// below safe to run over every operation.
template <class T>
void substitute(T&, const expr_map&)
{
}

template <class T>
void substitute(std::vector<T>& v, const expr_map& symbols);
template <class T>
void substitute(optional<T>& o, const expr_map& symbols);

template <class T>
void substitute(std::vector<T>& v, const expr_map& symbols)
{
    std::for_each(v.begin(), v.end(), [&](T& x) { substitute(x, symbols); });
}

template <class T>
void substitute(optional<T>& o, const expr_map& symbols)
{
    if(o.has_value())
        substitute(*o, symbols);
}

} // namespace detail

/**
 * Return a copy of an operation with every symbol in its reflected attributes replaced by the
 * size that symbol takes in this specialization. Operations carrying no symbols are returned
 * unchanged, so this is safe to apply to any operation.
 */
template <class T>
T substitute_symbols(T x, const symbol_map& symbols)
{
    auto exprs = detail::as_expr_map(symbols);
    reflect_each(x, [&](auto& member, auto&&...) { detail::substitute(member, exprs); });
    return x;
}

/**
 * Return a copy of a shape specialized to the given symbols. Unlike shape::to_static, which
 * needs a size for every symbol, this leaves symbols missing from the map alone, so a shape can
 * be specialized on one dimension while staying symbolic in the others. A dimension becomes
 * static once its expression resolves to a single size.
 */
MIGRAPHX_EXPORT shape substitute_symbols(shape s, const symbol_map& symbols);

/**
 * The sizes of a list of dimensions, if every one of them has settled on a single size. An
 * operator that takes a symbolic output shape opts in through a fully symbolic attribute, so once
 * substitution resolves that attribute the operator has to move over to its static form rather
 * than hold a mix of the two.
 */
MIGRAPHX_EXPORT optional<std::vector<std::size_t>>
fixed_lens(const std::vector<shape::dynamic_dimension>& dims);

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
