/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_UTILITY_OPERATORS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_UTILITY_OPERATORS_HPP

#include <migraphx/config.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/returns.hpp>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class X>
struct equality_comparable
{
    struct private_ops
    {
        template <class U, MIGRAPHX_REQUIRES(std::is_same<U, X>{})>
        static constexpr auto equal1(const U& x, const X& y) MIGRAPHX_RETURNS(x == y);

        template <class T, class U, MIGRAPHX_REQUIRES(std::is_same<T, X>{})>
        static constexpr auto equal2(const T& x, const U& y) MIGRAPHX_RETURNS(x.operator==(y));
    };

    friend constexpr auto operator!=(const X& x, const X& y)
    {
        return not(private_ops::equal1(x, y));
    }

    template <class U, class T>
    friend constexpr auto operator==(const U& x, const T& y)
        MIGRAPHX_RETURNS(private_ops::equal2(y, x));
    template <class U, class T>
    friend constexpr auto operator!=(const U& x, const T& y)
        MIGRAPHX_RETURNS(not(private_ops::equal2(y, x)));
};

template <class X>
struct less_than_comparable
{

    struct private_ops
    {

        template <class U, MIGRAPHX_REQUIRES(std::is_same<U, X>{})>
        static constexpr auto less1(const U& x, const X& y) MIGRAPHX_RETURNS(x < y);

        template <class T, class U, MIGRAPHX_REQUIRES(std::is_same<T, X>{})>
        static constexpr auto less2(const T& x, const U& y) MIGRAPHX_RETURNS(x.operator<(y));
        template <class T, class U, MIGRAPHX_REQUIRES(std::is_same<T, X>{})>
        static constexpr auto greater2(const T& x, const U& y) MIGRAPHX_RETURNS(x.operator>(y));
    };

    friend constexpr bool operator>(const X& x, const X& y) { return private_ops::less1(y, x); }
    friend constexpr bool operator<=(const X& x, const X& y)
    {
        return not(private_ops::less1(y, x));
    }
    friend constexpr bool operator>=(const X& x, const X& y)
    {
        return not(private_ops::less1(x, y));
    }

    template <class T, class U>
    friend constexpr auto operator<=(const T& x, const U& y)
        MIGRAPHX_RETURNS(not(private_ops::greater2(x, y)));
    template <class T, class U>
    friend constexpr auto operator>=(const T& x, const U& y)
        MIGRAPHX_RETURNS(not(private_ops::less2(x, y)));

    template <class U, class T>
    friend constexpr auto operator<(const U& x, const T& y)
        MIGRAPHX_RETURNS(private_ops::greater2(y, x));
    template <class U, class T>
    friend constexpr auto operator>(const U& x, const T& y)
        MIGRAPHX_RETURNS(private_ops::less2(y, x));
    template <class U, class T>
    friend constexpr auto operator<=(const U& x, const T& y)
        MIGRAPHX_RETURNS(not(private_ops::less2(y, x)));
    template <class U, class T>
    friend constexpr auto operator>=(const U& x, const T& y)
        MIGRAPHX_RETURNS(not(private_ops::greater2(y, x)));
};

template <class X>
struct equivalence
{
    using private_ops = typename less_than_comparable<X>::private_ops;

    friend constexpr auto operator==(const X& x, const X& y)
    {
        return not private_ops::less1(x, y) and not private_ops::less1(y, x);
    }

    template <class U, class T>
    friend constexpr auto operator==(const U& x, const T& y)
        MIGRAPHX_RETURNS(not private_ops::less2(y, x) and not private_ops::greater2(y, x));

    template <class T, class U>
    friend constexpr auto operator==(const T& x, const U& y)
        MIGRAPHX_RETURNS(not private_ops::less2(x, y) and not private_ops::greater2(x, y));
};

template <class X>
struct totally_ordered : equality_comparable<X>, less_than_comparable<X>
{
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_UTILITY_OPERATORS_HPP
