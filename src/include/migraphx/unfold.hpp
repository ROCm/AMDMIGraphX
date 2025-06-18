#ifndef MIGRAPHX_GUARD_MIGRAPHX_UNFOLD_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_UNFOLD_HPP

#include <migraphx/config.hpp>
#include <optional>
#include <iterator>
#include <utility>
#include <type_traits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class State, class F, class G>
struct unfold_range
{
    unfold_range(std::optional<State> pz, F pf, G pg)
        : z(std::move(pz)), f(std::move(pf)), g(std::move(pg))
    {
    }

    struct iterator
    {
        using reference         = decltype(std::declval<F>()(std::declval<State>()));
        using value_type        = std::decay_t<reference>;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::forward_iterator_tag;
        using pointer           = void;

        bool operator==(const iterator& other) const
        {
            return parent == other.parent and state == other.state;
        }
        bool operator!=(const iterator& other) const { return !(*this == other); }

        value_type operator*() const
        {
            // f is applied on the fly
            return parent->f(*state);
        }

        iterator& operator++()
        {
            state = parent->g(*state);
            return *this;
        }
        iterator operator++(int)
        {
            auto tmp = *this;
            ++(*this);
            return tmp;
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

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_UNFOLD_HPP
