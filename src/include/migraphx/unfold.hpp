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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_UNFOLD_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_UNFOLD_HPP

#include <migraphx/config.hpp>
#include <migraphx/iterator.hpp>
#include <optional>
#include <iterator>
#include <utility>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class State, class F, class G>
struct unfold_range : iterator_operators<unfold_range<State, F, G>>
{
    unfold_range(std::optional<State> pz, F pf, G pg)
        : z(std::move(pz)), f(std::move(pf)), g(std::move(pg))
    {
    }

    struct iterator : iterator_operators<iterator>
    {
        using reference         = decltype(std::declval<F>()(std::declval<State>()));
        using value_type        = std::decay_t<reference>;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        using pointer           = std::add_pointer_t<std::remove_reference_t<reference>>;

        iterator() = default;

        iterator(const unfold_range* pparent, std::optional<State> pstate)
            : parent(pparent), state(std::move(pstate))
        {
        }

        reference operator*() const { return parent->f(*state); }

        template <class U>
        static void increment(U& x)
        {
            x.state = x.parent->g(*x.state);
        }

        template <class U, class V>
        static auto equal(const U& x, const V& y)
        {
            return x.parent == y.parent and x.state == y.state;
        }

        const unfold_range* parent = nullptr;
        std::optional<State> state = std::nullopt;
    };

    iterator begin() const { return iterator{this, z}; }
    iterator end() const { return iterator{this, std::nullopt}; }

    private:
    std::optional<State> z;
    F f;
    G g;
};

/**
 * @brief Returns a range which generates a sequence by repeatedly applying a step function and
 * projecting each state.
 *
 * The `unfold` function creates a range starting from an initial state `z`. At each step, the range
 * yields `f(state)` as the sequence element, and advances the state by applying `g(state)`. If
 * `g(state)` returns `std::nullopt`, the sequence ends.
 *
 * - The range's value type is the result of `f(State)`.
 * - The sequence begins with the initial state `z`.
 * - At each iteration, the current element is `f(state)`, and the next state is computed by
 * `g(state)`.
 * - When `g(state)` returns `std::nullopt`, the range ends (the end iterator is reached).
 *
 * This function is analogous to Haskell's `unfoldr`, but instead of returning a pair `(a, b)` or
 * `Nothing`, the generator function `g` here is only responsible for producing the next state: on
 * each step, `f(state)` produces the value, and `g(state)` produces the next state or
 * `std::nullopt` to stop.
 *
 * Example:
 *   auto rng = unfold(1, [](int x) { return x; }, [](int x) -> std::optional<int> {
 *       if (x < 5) return x+1;
 *       return std::nullopt;
 *   });
 *   // rng yields 1, 2, 3, 4, 5
 *
 * @param z The initial state.
 * @param f Function to apply to the state to generate the value for each element.
 * @param g Function to step to the next state. Returns std::optional<State>; if std::nullopt is
 * returned, iteration stops.
 * @return An input range whose iterator yields values of type `decltype(f(z))`.
 */
template <class State, class F, class G>
auto unfold(State z, F f, G g)
{
    return unfold_range<State, F, G>(std::move(z), std::move(f), std::move(g));
}

template <class State, class F, class G>
auto unfold(std::nullopt_t z, F f, G g)
{
    return unfold_range<State, F, G>(std::move(z), std::move(f), std::move(g));
}

template <class State, class G>
auto unfold(State z, G g)
{
    return unfold(std::move(z), [](const auto& x) -> const auto& { return x; }, std::move(g));
}

template <class State, class G>
auto unfold(std::nullopt_t z, G g)
{
    return unfold<State>(z, [](const auto& x) -> const auto& { return x; }, std::move(g));
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_UNFOLD_HPP
