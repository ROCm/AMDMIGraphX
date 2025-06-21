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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_TRANSFORM_VIEW_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_TRANSFORM_VIEW_HPP

#include <migraphx/config.hpp>
#include <migraphx/iterator.hpp>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace views {

template <class Range, class F>
struct transform_view
{

    constexpr transform_view(Range& prng, F pf) : rng(&prng), f(std::move(pf)) {}

    struct iterator : iterator_operators<iterator>
    {
        using underlying_iterator = decltype(std::begin(std::declval<Range&>()));
        using reference           = decltype(std::declval<const F>()(
            std::declval<typename std::iterator_traits<underlying_iterator>::reference>()));
        using value_type          = std::decay_t<reference>;

        using iterator_category =
            typename std::iterator_traits<underlying_iterator>::iterator_category;
        using difference_type = typename std::iterator_traits<underlying_iterator>::difference_type;
        using pointer         = std::add_pointer_t<std::remove_reference_t<reference>>;

        constexpr iterator() = default;

        constexpr iterator(const transform_view* pparent, underlying_iterator it)
            : parent(pparent), current(it)
        {
        }

        constexpr reference operator*() const { return parent->f(*current); }

        template <class U>
        static auto increment(U& x) -> decltype(++x.current)
        {
            return ++x.current;
        }

        template <class U>
        static auto decrement(U& x) -> decltype(--x.current)
        {
            return --x.current;
        }

        template <class U, class I>
        static auto advance(U& x, I n) -> decltype(x.current += n)
        {
            return x.current += n;
        }

        template <class U, class V>
        static auto distance(const U& x, const V& y) -> decltype(x.parent == y.parent,
                                                                 y.current - x.current)
        {
            assert(x.parent == y.parent);
            return y.current - x.current;
        }

        template <class U, class V>
        static auto equal(const U& x, const V& y) -> decltype(x.current == y.current)
        {
            return x.parent == y.parent and x.current == y.current;
        }

        private:
        const transform_view* parent = nullptr;
        underlying_iterator current{};
    };

    constexpr iterator begin() const { return {this, std::begin(*rng)}; }
    constexpr iterator end() const { return {this, std::end(*rng)}; }

    friend constexpr bool operator==(const transform_view& a, const transform_view& b)
    {
        return std::equal(a.begin(), a.end(), b.begin(), b.end());
    }

    friend constexpr bool operator!=(const transform_view& a, const transform_view& b)
    {
        return not(a == b);
    }

    friend constexpr bool operator<(const transform_view& a, const transform_view& b)
    {
        return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end());
    }

    friend constexpr bool operator>(const transform_view& a, const transform_view& b)
    {
        return b < a;
    }

    friend constexpr bool operator<=(const transform_view& a, const transform_view& b)
    {
        return not(b < a);
    }

    friend constexpr bool operator>=(const transform_view& a, const transform_view& b)
    {
        return not(a < b);
    }

    private:
    Range* rng = nullptr;
    F f;
};

// helper for type deduction
template <class Range, class F>
auto transform(Range& rng, F f)
{
    return transform_view<Range, F>(rng, std::move(f));
}

} // namespace views
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_TRANSFORM_VIEW_HPP
