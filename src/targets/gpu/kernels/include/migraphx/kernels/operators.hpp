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
 *
 */
#ifndef MIGRAPHX_GUARD_KERNELS_OPERATORS_HPP
#define MIGRAPHX_GUARD_KERNELS_OPERATORS_HPP

#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/type_traits.hpp>

namespace migraphx {

// NOLINTNEXTLINE
#define MIGRAPHX_DEFINE_OPERATOR(op, expr)                                                  \
    template <class U>                                                                      \
    friend constexpr auto operator op(const T& x, const U& y) MIGRAPHX_RETURNS(expr);       \
    template <class U, class V, MIGRAPHX_REQUIRES(not is_same<T, U>{} and is_same<V, T>{})> \
    friend constexpr auto operator op(const U& x, const V& y) MIGRAPHX_RETURNS(expr)

template <class T>
struct equality_comparable
{
    MIGRAPHX_DEFINE_OPERATOR(!=, not(x == y));
};

template <class T>
struct less_than_comparable
{
    MIGRAPHX_DEFINE_OPERATOR(>, (y < x));
    MIGRAPHX_DEFINE_OPERATOR(<=, not(y < x));
    MIGRAPHX_DEFINE_OPERATOR(>=, not(x < y));
};

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_OPERATORS_HPP
