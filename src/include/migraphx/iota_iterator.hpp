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
#ifndef MIGRAPHX_GUARD_RTGLIB_IOTA_ITERATOR_HPP
#define MIGRAPHX_GUARD_RTGLIB_IOTA_ITERATOR_HPP

#include <migraphx/config.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/iterator.hpp>
#include <iterator>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class F, class Iterator = std::ptrdiff_t>
struct basic_iota_iterator : iterator_operators<basic_iota_iterator<F, Iterator>>,
                             iterator_types<decltype(std::declval<F>()(std::declval<Iterator>())),
                                            std::random_access_iterator_tag>
{
    Iterator index;
    F f;

    using reference = decltype(std::declval<F>()(std::declval<Iterator>()));

    // cppcheck-suppress uninitMemberVar
    constexpr basic_iota_iterator() = default;

    template <class... Ts>
    constexpr basic_iota_iterator(Iterator i, Ts&&... xs) : index(i), f{std::forward<Ts>(xs)...}
    {
    }

    constexpr basic_iota_iterator::reference operator*() const { return f(index); }

    template <class U>
    static constexpr auto increment(U& x) -> decltype(++x.index)
    {
        return ++x.index;
    }

    template <class U>
    static constexpr auto decrement(U& x) -> decltype(--x.index)
    {
        return --x.index;
    }

    template <class U, class I>
    static constexpr auto advance(U& x, I n) -> decltype(x.index += n)
    {
        return x.index += n;
    }

    template <class U, class V>
    static constexpr auto distance(const U& x, const V& y) -> decltype(y.index - x.index)
    {
        return y.index - x.index;
    }

    template <class U, class V>
    static constexpr auto equal(const U& x, const V& y) -> decltype(x.index == y.index)
    {
        return x.index == y.index;
    }

    template <class Stream>
    friend Stream& operator<<(Stream& s, const basic_iota_iterator& x)
    {
        return s << x.index;
    }
};

template <class T, class F>
basic_iota_iterator<F, T> make_basic_iota_iterator(T x, F f)
{
    return {x, f};
}

using iota_iterator = basic_iota_iterator<id>;

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
