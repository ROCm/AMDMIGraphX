#ifndef MIGRAPH_GUARD_RTGLIB_FUNCTIONAL_HPP
#define MIGRAPH_GUARD_RTGLIB_FUNCTIONAL_HPP

#include <utility>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

struct swallow
{
    template <class... Ts>
    constexpr swallow(Ts&&...)
    {
    }
};

namespace detail {

template <class R, class F>
struct fix_f
{
    F f;

    template <class... Ts>
    R operator()(Ts&&... xs) const
    {
        return f(*this, std::forward<Ts>(xs)...);
    }
};

template <std::size_t...>
struct seq
{
    using type = seq;
};

template <class, class>
struct merge_seq;

template <std::size_t... Xs, std::size_t... Ys>
struct merge_seq<seq<Xs...>, seq<Ys...>> : seq<Xs..., (sizeof...(Xs) + Ys)...>
{
};

template <std::size_t N>
struct gens : merge_seq<typename gens<N / 2>::type, typename gens<N - N / 2>::type>
{
};

template <>
struct gens<0> : seq<>
{
};
template <>
struct gens<1> : seq<0>
{
};

template <class F, std::size_t... Ns>
constexpr void repeat_c_impl(F f, seq<Ns...>)
{
    swallow{(f(std::integral_constant<std::size_t, Ns>{}), 0)...};
}

template <class F, std::size_t... Ns>
constexpr auto sequence_c_impl(F&& f, seq<Ns...>)
{
    return f(std::integral_constant<std::size_t, Ns>{}...);
}

} // namespace detail

template <std::size_t N, class F>
constexpr void repeat_c(F f)
{
    detail::repeat_c_impl(f, detail::gens<N>{});
}

template <std::size_t N, class F>
constexpr auto sequence_c(F&& f)
{
    return detail::sequence_c_impl(f, detail::gens<N>{});
}

template <class F, class... Ts>
constexpr void each_args(F f, Ts&&... xs)
{
    swallow{(f(std::forward<Ts>(xs)), 0)...};
}

template <class F>
constexpr void each_args(F)
{
}

/// Implements a fix-point combinator
template <class R, class F>
detail::fix_f<R, F> fix(F f)
{
    return {f};
}

template <class F>
auto fix(F f)
{
    return fix<void>(f);
}

template <class... Ts>
auto pack(Ts... xs)
{
    return [=](auto f) { return f(xs...); };
}

template <class F, class T>
auto fold_impl(F&&, T&& x)
{
    return x;
}

template <class F, class T, class U, class... Ts>
auto fold_impl(F&& f, T&& x, U&& y, Ts&&... xs)
{
    return fold_impl(f, f(std::forward<T>(x), std::forward<U>(y)), std::forward<Ts>(xs)...);
}

template <class F>
auto fold(F f)
{
    return [=](auto&&... xs) { return fold_impl(f, std::forward<decltype(xs)>(xs)...); };
}

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
