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
#include <migraphx/requires.hpp>
#include <migraphx/utility_operators.hpp>
#include <algorithm>
#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace views {

// zip_view is a range adaptor that takes one or more ranges, and produces a
// view whose ith element is a tuple-like value consisting of the ith
// elements of all ranges. The size of produced view is the minimum of sizes
// of all adapted ranges.
template <class... Ranges>
struct zip_view : totally_ordered<zip_view<Ranges...>>
{
    constexpr explicit zip_view(Ranges&... prngs) : rng(std::addressof(prngs)...) {}

    template <class... BaseIterators>
    struct iterator : iterator_operators<iterator<BaseIterators...>>
    {
        using reference  = std::tuple<typename std::iterator_traits<BaseIterators>::reference...>;
        using value_type = std::tuple<typename std::iterator_traits<BaseIterators>::value_type...>;

        // Weakest underlying category governs the zip (each tag derives from the weaker one).
        using common_category =
            std::common_type_t<typename std::iterator_traits<BaseIterators>::iterator_category...>;
        static constexpr bool is_random_access =
            std::is_base_of<std::random_access_iterator_tag, common_category>{};
        static constexpr bool is_bidirectional =
            std::is_base_of<std::bidirectional_iterator_tag, common_category>{};

        // Bidirectional iterators become forward iterators because
        // decrementing from ragged ends doesnt work correctly without
        // needing a larger overhead/complexity.
        using iterator_category = std::conditional_t<is_bidirectional and not is_random_access,
                                                     std::forward_iterator_tag,
                                                     common_category>;
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

        // Only random-access zips decrement (keyed off U so the SFINAE is dependent).
        template <class U,
                  MIGRAPHX_REQUIRES(std::is_base_of<std::random_access_iterator_tag,
                                                    typename U::iterator_category>{})>
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
            // Random-access only: iterators are in lockstep, so any one pair gives the distance.
            // Ill-formed for weaker iterators, which removes operator-/< (a forward zip has none).
            return std::get<0>(y.current) - std::get<0>(x.current);
        }

        template <class U, class V>
        static bool equal(const U& x, const V& y)
        {
            if constexpr(is_random_access)
            {
                // Truncated end is in lockstep, so plain tuple equality.
                return x.current == y.current;
            }
            else
            {
                // Ragged end, forward-only: match if any pair matches (stops at the shortest
                // range). No decrement means no mixed-offset iterators, so no false matches.
                return migraphx::unpack(
                    [&](const auto&... xits) {
                        return migraphx::unpack(
                            [&](const auto&... yits) { return ((xits == yits) or ...); },
                            y.current);
                    },
                    x.current);
            }
        }

        private:
        std::tuple<BaseIterators...> current{};
    };

    template <class... BaseIterators>
    static constexpr iterator<BaseIterators...> make_iterator(std::tuple<BaseIterators...> its)
    {
        return iterator<BaseIterators...>{std::move(its)};
    }

    // Random-access end: advance begins by the shortest length (O(1)) so end is in lockstep.
    template <class Its, class Ends>
    static auto truncated_end(Its its, const Ends& ends)
    {
        const auto n = migraphx::unpack(
            [&](auto&... bs) {
                return migraphx::unpack(
                    [&](auto&... es) {
                        using diff = std::common_type_t<typename std::iterator_traits<
                            std::decay_t<decltype(bs)>>::difference_type...>;
                        return std::min({static_cast<diff>(std::distance(bs, es))...});
                    },
                    ends);
            },
            its);
        migraphx::unpack([&](auto&... bs) { (std::advance(bs, n), ...); }, its);
        return make_iterator(std::move(its));
    }

    // Truncated (in-lockstep) end for random-access ranges, ragged end otherwise.
    template <class... Is, class Ends>
    static auto make_end_iterator(std::tuple<Is...> begins, Ends ends)
    {
        using category =
            std::common_type_t<typename std::iterator_traits<Is>::iterator_category...>;
        if constexpr(std::is_base_of<std::random_access_iterator_tag, category>{})
            return truncated_end(std::move(begins), ends);
        else
            return make_iterator(std::move(ends));
    }

    auto begin()
    {
        return migraphx::unpack(
            [](auto*... rs) { return make_iterator(std::make_tuple(std::begin(*rs)...)); }, rng);
    }
    auto end()
    {
        return migraphx::unpack(
            [](auto*... rs) {
                return make_end_iterator(std::make_tuple(std::begin(*rs)...),
                                         std::make_tuple(std::end(*rs)...));
            },
            rng);
    }

    // rng is plain pointers, so const doesn't propagate; view ranges as const for const iterators.
    auto begin() const
    {
        return migraphx::unpack(
            [](auto*... rs) {
                return make_iterator(std::make_tuple(std::begin(std::as_const(*rs))...));
            },
            rng);
    }
    auto end() const
    {
        return migraphx::unpack(
            [](auto*... rs) {
                return make_end_iterator(std::make_tuple(std::begin(std::as_const(*rs))...),
                                         std::make_tuple(std::end(std::as_const(*rs))...));
            },
            rng);
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
