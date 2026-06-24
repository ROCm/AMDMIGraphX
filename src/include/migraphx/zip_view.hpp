/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_ZIP_VIEW_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_ZIP_VIEW_HPP

#include <migraphx/config.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/iterator.hpp>
#include <migraphx/utility_operators.hpp>
#include <algorithm>
#include <array>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace views {

template <class... Ranges>
struct zip_view : totally_ordered<zip_view<Ranges...>>
{
    constexpr explicit zip_view(Ranges&... prngs) : rng(std::addressof(prngs)...) {}

    template <class... BaseIterators>
    struct iterator : iterator_operators<iterator<BaseIterators...>>
    {
        using reference  = std::tuple<typename std::iterator_traits<BaseIterators>::reference...>;
        using value_type = std::tuple<typename std::iterator_traits<BaseIterators>::value_type...>;

        // The weakest category among the underlying iterators governs the zip, and
        // std::common_type of the tags yields it (each tag derives from the weaker one).
        using iterator_category =
            std::common_type_t<typename std::iterator_traits<BaseIterators>::iterator_category...>;
        using difference_type =
            std::common_type_t<typename std::iterator_traits<BaseIterators>::difference_type...>;
        using pointer = std::add_pointer_t<std::remove_reference_t<reference>>;

        constexpr iterator() = default;

        constexpr explicit iterator(std::tuple<BaseIterators...> its) : current(std::move(its)) {}

        reference operator*() const
        {
            return migraphx::unpack([](auto&... its) { return reference{*its...}; }, current);
        }

        template <class U>
        static void increment(U& x)
        {
            migraphx::unpack([](auto&... its) { (++its, ...); }, x.current);
        }

        template <class U>
        static void decrement(U& x)
        {
            migraphx::unpack([](auto&... its) { (--its, ...); }, x.current);
        }

        template <class U, class I>
        static void advance(U& x, I n)
        {
            migraphx::unpack([&](auto&... its) { ((its += n), ...); }, x.current);
        }

        template <class U, class V>
        static auto distance(const U& x, const V& y)
        {
            // A zip stops at its shortest range, so the smallest-magnitude per-iterator
            // distance is the number of valid steps between the two iterators.
            return migraphx::unpack(
                [&](const auto&... xits) {
                    return migraphx::unpack(
                        [&](const auto&... yits) {
                            using diff = std::common_type_t<decltype(yits - xits)...>;
                            const std::array<diff, sizeof...(yits)> dists = {
                                static_cast<diff>(yits - xits)...};
                            return *std::min_element(
                                dists.begin(), dists.end(), [](diff a, diff b) {
                                    return (a < 0 ? -a : a) < (b < 0 ? -b : b);
                                });
                        },
                        y.current);
                },
                x.current);
        }

        template <class U, class V>
        static bool equal(const U& x, const V& y)
        {
            // Equal when any underlying iterator matches so that iteration stops as soon
            // as the shortest underlying range is exhausted.
            return migraphx::unpack(
                [&](const auto&... xits) {
                    return migraphx::unpack(
                        [&](const auto&... yits) { return ((xits == yits) or ...); }, y.current);
                },
                x.current);
        }

        private:
        std::tuple<BaseIterators...> current{};
    };

    template <class... BaseIterators>
    static constexpr iterator<BaseIterators...> make_iterator(std::tuple<BaseIterators...> its)
    {
        return iterator<BaseIterators...>{std::move(its)};
    }

    auto begin()
    {
        return migraphx::unpack(
            [](auto*... rs) { return make_iterator(std::make_tuple(std::begin(*rs)...)); }, rng);
    }
    auto end()
    {
        return migraphx::unpack(
            [](auto*... rs) { return make_iterator(std::make_tuple(std::end(*rs)...)); }, rng);
    }

    auto begin() const
    {
        return migraphx::unpack(
            [](auto*... rs) { return make_iterator(std::make_tuple(std::begin(*rs)...)); }, rng);
    }
    auto end() const
    {
        return migraphx::unpack(
            [](auto*... rs) { return make_iterator(std::make_tuple(std::end(*rs)...)); }, rng);
    }

    template <class... Ts>
    bool operator==(const zip_view<Ts...>& b) const
    {
        return std::equal(this->begin(), this->end(), b.begin(), b.end());
    }

    template <class... Ts>
    bool operator<(const zip_view<Ts...>& b) const
    {
        return std::lexicographical_compare(this->begin(), this->end(), b.begin(), b.end());
    }

    private:
    std::tuple<Ranges*...> rng;
};

template <class... Ranges>
auto zip(Ranges&... rngs)
{
    return zip_view<Ranges...>(rngs...);
}

} // namespace views
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_ZIP_VIEW_HPP
