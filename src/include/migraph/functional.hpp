#ifndef MIGRAPH_GUARD_RTGLIB_FUNCTIONAL_HPP
#define MIGRAPH_GUARD_RTGLIB_FUNCTIONAL_HPP

#include <utility>

namespace migraph {

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

} // namespace detail

template <std::size_t N, class F>
constexpr void repeat_c(F f)
{
    detail::repeat_c_impl(f, detail::gens<N>{});
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

} // namespace migraph

#endif
