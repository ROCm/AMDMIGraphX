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
 */
#ifndef MIGRAPHX_GUARD_MIGRAPHX_ITERATOR_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_ITERATOR_HPP

#include <migraphx/config.hpp>
#include <migraphx/rank.hpp>
#include <migraphx/requires.hpp>
#include <cassert>
#include <memory>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Iterator, class EndIterator>
auto is_end(rank<2>, Iterator it, EndIterator) -> decltype(not it._M_dereferenceable())
{
    return not it._M_dereferenceable();
}

template <class Iterator, class EndIterator>
auto is_end(rank<1>, Iterator it, EndIterator last)
{
    return it == last;
}

template <class Iterator, class EndIterator>
bool is_end(Iterator it, EndIterator last)
{
    return is_end(rank<2>{}, it, last);
}

template <class Iterator>
auto* iterator_address(rank<0>, Iterator it)
{
    return std::addressof(*it);
}

template <class Iterator>
auto iterator_address(rank<1>, Iterator it) -> decltype(it._M_dereferenceable()
                                                            ? std::addressof(*it)
                                                            : nullptr)
{
    return it._M_dereferenceable() ? std::addressof(*it) : nullptr;
}

template <class Iterator>
auto iterator_address(rank<1>,
                      Iterator it) -> decltype(std::addressof(it._Unwrapped()._Ptr->_Myval),
                                               std::addressof(*it))
{
    return it._Unwrapped()._Ptr ? std::addressof(it._Unwrapped()._Ptr->_Myval) : nullptr;
}

template <class Iterator>
auto* iterator_address(Iterator it)
{
    return iterator_address(rank<1>{}, it);
}

template <class T>
struct arrow_proxy
{
    T x;
    constexpr auto operator->() const { return std::addressof(x); }
};

template <class T>
arrow_proxy(T&&) -> arrow_proxy<T>;

template <class T>
struct iterator_operators
{
    // Core operators

    // Increment
    template <class U, MIGRAPHX_REQUIRES(std::is_same<U, T>{})>
    friend constexpr auto operator++(U& x) -> decltype(U::increment(x), std::declval<U&>())
    {
        U::increment(x);
        return x;
    }

    // Decrement
    template <class U, MIGRAPHX_REQUIRES(std::is_same<U, T>{})>
    friend constexpr auto operator--(U& x) -> decltype(U::decrement(x), std::declval<U&>())
    {
        U::decrement(x);
        return x;
    }

    // Advance
    template <class U, class I, MIGRAPHX_REQUIRES(std::is_same<U, T>{})>
    friend constexpr auto operator+=(U& x, I n) -> decltype(U::increment(x), std::declval<U&>())
    {
        U::advance(x, n);
        return x;
    }

    // Distance
    template <class U, class Self = T>
    friend constexpr auto operator-(const U& x, const T& y) -> decltype(Self::distance(y, x))
    {
        return Self::distance(y, x);
    }

    // Equal
    template <class U, class Self = T>
    friend constexpr auto operator==(const T& x, const U& y) -> decltype(Self::equal(x, y))
    {
        return Self::equal(x, y);
    }

    template <class U>
    friend constexpr auto operator<(const T& x,
                                    const U& y) -> decltype(static_cast<bool>((x - y) < 0))
    {
        return static_cast<bool>((x - y) < 0);
    }

    template <class U>
    friend constexpr auto operator<=(const T& x,
                                     const U& y) -> decltype(not static_cast<bool>(x > y))
    {
        return not static_cast<bool>(x > y);
    }

    template <class U>
    friend constexpr auto operator>=(const T& x,
                                     const U& y) -> decltype(not static_cast<bool>(x < y))
    {
        return not static_cast<bool>(x < y);
    }

    template <class U>
    friend constexpr auto operator>(const T& x, const U& y) -> decltype(y < x)
    {
        return y < x;
    }

    template <class U>
    friend constexpr auto operator!=(const T& x,
                                     const U& y) -> decltype(not static_cast<bool>(x == y))
    {
        return not static_cast<bool>(x == y);
    }

    template <class U>
    friend constexpr auto operator+(T lhs, const U& rhs) -> decltype(T(lhs += rhs))
    {
        return lhs += rhs;
    }

    template <class U, MIGRAPHX_REQUIRES(not std::is_convertible<U, T>{})>
    friend constexpr auto operator+(const U& lhs, T rhs) -> decltype(T(rhs += lhs))
    {
        return rhs += lhs;
    }

    template <class U>
    friend constexpr auto operator-=(T& lhs, const U& rhs) -> decltype(lhs += -rhs)
    {
        return lhs += -rhs;
    }

    template <class U = T>
    friend constexpr auto operator++(U& x, int) -> decltype(T(++std::declval<U>()))
    {
        T nrv(x);
        ++x;
        return nrv;
    }

    template <class U = T>
    friend constexpr auto operator--(U& x, int) -> decltype(T(--std::declval<U>()))
    {
        T nrv(x);
        --x;
        return nrv;
    }

    template <class I, class U = T>
    constexpr auto operator[](I n) const -> decltype(*(static_cast<const U&>(*this) + n))
    {
        return *(static_cast<const U&>(*this) + n);
    }

    template <class U = T>
    constexpr auto operator->() const -> decltype(arrow_proxy{*static_cast<const U&>(*this)})
    {
        return arrow_proxy{*static_cast<const U&>(*this)};
    }
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_ITERATOR_HPP
