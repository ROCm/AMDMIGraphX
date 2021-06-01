#ifndef MIGRAPHX_GUARD_KERNELS_FUNCTIONAL_HPP
#define MIGRAPHX_GUARD_KERNELS_FUNCTIONAL_HPP

#include <migraphx/kernels/array.hpp>

namespace migraphx {

struct swallow
{
    template <class... Ts>
    constexpr swallow(Ts&&...)
    {
    }
};

template<index_int>
using ignore = swallow;

namespace detail {

template <index_int...>
struct seq
{
    using type = seq;
};

template <class, class>
struct merge_seq;

template <index_int... Xs, index_int... Ys>
struct merge_seq<seq<Xs...>, seq<Ys...>> : seq<Xs..., (sizeof...(Xs) + Ys)...>
{
};

template <index_int N>
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

template <class F, index_int... Ns>
constexpr auto sequence_c_impl(F&& f, seq<Ns...>)
{
    return f(index_constant<Ns>{}...);
}

template<index_int... N>
constexpr auto args_at(seq<N...>)
{
    return [](ignore<N>..., auto x, auto...) {
        return x;
    };
}

} // namespace detail

template <index_int N, class F>
constexpr auto sequence_c(F&& f)
{
    return detail::sequence_c_impl(f, detail::gens<N>{});
}

template <class IntegerConstant, class F>
constexpr auto sequence(IntegerConstant ic, F&& f)
{
    return sequence_c<ic>(f);
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

template <class... Ts>
auto pack(Ts... xs)
{
    return [=](auto f) { return f(xs...); };
}

template<index_int N>
constexpr auto arg_c()
{
    return [](auto... xs) {
        return detail::args_at(detail::gens<N>{})(xs...);
    };
}

template<class IntegralConstant>
constexpr auto arg(IntegralConstant ic)
{
    return arg_c<ic>();
}

template <class... Ts>
constexpr auto rotate_last(Ts... xs)
{
    return [=](auto&& f) {
        return sequence_c<sizeof...(Ts)>([&](auto... is) { 
            constexpr auto size = sizeof...(is);
            return f(arg_c<(is + size - 1) % size>()(xs...)...);
        });
    };
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_FUNCTIONAL_HPP
