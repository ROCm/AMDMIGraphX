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

template <index_int>
using ignore = swallow;

namespace detail {

template <class R>
struct eval_helper
{
    R result;

    template <class F, class... Ts>
    constexpr eval_helper(const F& f, Ts&&... xs) : result(f(static_cast<Ts>(xs)...))
    {
    }
};

template <>
struct eval_helper<void>
{
    int result;
    template <class F, class... Ts>
    constexpr eval_helper(const F& f, Ts&&... xs) : result((f(static_cast<Ts>(xs)...), 0))
    {
    }
};

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

template <index_int... N>
constexpr auto args_at(seq<N...>)
{
    return [](ignore<N>..., auto x, auto...) { return x; };
}

} // namespace detail

template <class T>
constexpr auto always(T x)
{
    return [=](auto&&...) { return x; };
}

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

template <class F, class G>
constexpr auto by(F f, G g)
{
    return [=](auto... xs) {
        return detail::eval_helper<decltype(g(f(xs)...))>{g, f(xs)...}.result;
    };
}

template <class F>
constexpr auto by(F f)
{
    return by([=](auto x) { return (f(x), 0); }, always(0));
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

template <index_int N>
constexpr auto arg_c()
{
    return [](auto... xs) { return detail::args_at(detail::gens<N>{})(xs...); };
}

template <class IntegralConstant>
constexpr auto arg(IntegralConstant ic)
{
    return arg_c<ic>();
}

inline constexpr auto rotate_last()
{
    return [](auto... xs) {
        return [=](auto&& f) {
            return sequence_c<sizeof...(xs)>([&](auto... is) {
                constexpr auto size = sizeof...(is);
                return f(arg_c<(is + size - 1) % size>()(xs...)...);
            });
        };
    };
}

template <class F>
constexpr auto transform_args(F f)
{
    return [=](auto... xs) {
        return [=](auto g) { return f(xs...)([&](auto... ys) { return g(ys...); }); };
    };
}

template <class F, class... Fs>
constexpr auto transform_args(F f, Fs... fs)
{
    return [=](auto... xs) { return transform_args(f)(xs...)(transform_args(fs...)); };
}

// NOLINTNEXTLINE
#define MIGRAPHX_LIFT(...) \
    ([](auto&&... xs) { return (__VA_ARGS__)(static_cast<decltype(xs)>(xs)...); })

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_FUNCTIONAL_HPP
