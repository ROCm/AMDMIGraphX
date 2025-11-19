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
#include <migraphx/utility_operators.hpp>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace views {

template <class Range, class F>
struct transform_view : totally_ordered<transform_view<Range, F>>
{

    constexpr transform_view(Range& prng, F pf) : rng(&prng), f(std::move(pf)) {}

    template <class BaseIterator>
    struct iterator : iterator_operators<iterator<BaseIterator>>
    {
        using reference  = decltype(std::declval<const F>()(
            std::declval<typename std::iterator_traits<BaseIterator>::reference>()));
        using value_type = std::decay_t<reference>;

        using iterator_category = typename std::iterator_traits<BaseIterator>::iterator_category;
        using difference_type   = typename std::iterator_traits<BaseIterator>::difference_type;
        using pointer           = std::add_pointer_t<std::remove_reference_t<reference>>;

        constexpr iterator() = default;

        constexpr iterator(const transform_view* pparent, BaseIterator it)
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
        static auto equal(const U& x,
                          const V& y) -> decltype(x.parent == y.parent and x.current == y.current)
        {
            return x.parent == y.parent and x.current == y.current;
        }

        private:
        const transform_view* parent = nullptr;
        BaseIterator current{};
    };

    template <class BaseIterator>
    static constexpr iterator<BaseIterator> make_iterator(const transform_view* v, BaseIterator it)
    {
        return {v, it};
    }

    constexpr auto begin() const { return make_iterator(this, std::begin(base())); }
    constexpr auto end() const { return make_iterator(this, std::end(base())); }

    constexpr auto begin() { return make_iterator(this, std::begin(base())); }
    constexpr auto end() { return make_iterator(this, std::end(base())); }

    constexpr Range& base() { return *rng; }
    constexpr const Range& base() const { return *rng; }

    template <class... Ts>
    constexpr bool operator==(const transform_view<Ts...>& b) const
    {
        return std::equal(this->begin(), this->end(), b.begin(), b.end());
    }

    template <class... Ts>
    constexpr bool operator<(const transform_view<Ts...>& b) const
    {
        return std::lexicographical_compare(this->begin(), this->end(), b.begin(), b.end());
    }

    private:
    Range* rng = nullptr;
    F f;
};

template <class Range, class F>
auto transform(Range& rng, F f)
{
    return transform_view<Range, F>(rng, std::move(f));
}

} // namespace views
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_TRANSFORM_VIEW_HPP
