/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
 */
#ifndef MIGRAPHX_GUARD_KERNELS_FUNCTIONAL_HPP
#define MIGRAPHX_GUARD_KERNELS_FUNCTIONAL_HPP

#include <migraphx/kernels/integral_constant.hpp>

// NOLINTNEXTLINE
#define MIGRAPHX_RETURNS(...) \
    ->decltype(__VA_ARGS__) { return __VA_ARGS__; }

// NOLINTNEXTLINE
#define MIGRAPHX_LIFT(...)                           \
    [](auto&&... private_lisft_xs) MIGRAPHX_RETURNS( \
        (__VA_ARGS__)(static_cast<decltype(private_lisft_xs)>(private_lisft_xs)...))

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

template <class... Fs>
struct overloaded : Fs...
{
    using Fs::operator()...;
    overloaded(Fs... fs) : Fs(fs)... {}
};

template <class... Fs>
overloaded<Fs...> overload(Fs... fs)
{
    return {fs...};
}

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
    swallow{(f(static_cast<Ts&&>(xs)), 0)...};
}

template <class F>
constexpr void each_args(F)
{
}

template <class F, class T>
constexpr auto fold_impl(F&&, T&& x)
{
    return static_cast<T&&>(x);
}

template <class F, class T, class U, class... Ts>
constexpr auto fold_impl(F&& f, T&& x, U&& y, Ts&&... xs)
{
    return fold_impl(f, f(static_cast<T&&>(x), static_cast<U&&>(y)), static_cast<Ts&&>(xs)...);
}

template <class F>
constexpr auto fold(F f)
{
    return [=](auto&&... xs) { return fold_impl(f, static_cast<decltype(xs)&&>(xs)...); };
}

template <class... Ts>
constexpr auto pack(Ts... xs)
{
    return [=](auto f) { return f(xs...); };
}

template <class G, class F>
constexpr auto join(G g, F f)
{
    return f([=](auto... xs) { return g(xs...); });
}

template <class G, class F, class... Fs>
constexpr auto join(G g, F f, Fs... fs)
{
    return f([=](auto... xs) { return join([=](auto... ys) { return g(xs..., ys...); }, fs...); });
}

template <class Compare, class P1, class P2>
constexpr auto pack_compare(Compare compare, P1 p1, P2 p2)
{
    return p1([&](auto... xs) {
        return p2([&](auto... ys) {
            auto c = [&](auto x, auto y) -> int {
                if(compare(x, y))
                    return 1;
                else if(compare(y, x))
                    return -1;
                else
                    return 0;
            };
            return fold([](auto x, auto y) { return x ? x : y; })(c(xs, ys)..., 0);
        });
    });
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

template <class F>
constexpr auto make_transform(F f)
{
    return [=](auto... xs) { return [=](auto g) { return f(g, xs...); }; };
}

// An arg transformation takes the arguments and then a function to take the new arguments:
//     transform(xs...)([](auto... ys) { ... })
// The transform_args function takes a list of transformations and continually applies them
template <class F>
constexpr auto transform_args(F f)
{
    return f;
}

template <class F, class... Fs>
constexpr auto transform_args(F f, Fs... fs)
{
    return make_transform([=](auto g, auto... xs) {
        return f(xs...)([=](auto... ys) { return transform_args(fs...)(ys...)(g); });
    });
}

// identity transform
inline constexpr auto transform_args()
{
    return make_transform([](auto f, auto... xs) { return f(xs...); });
}

// Rotate the first argument to the last argument
inline constexpr auto rotate_last()
{
    return make_transform([](auto f, auto... xs) {
        return sequence_c<sizeof...(xs)>([&](auto... is) {
            constexpr auto size = sizeof...(is);
            return f(arg_c<(is + size - 1) % size>()(xs...)...);
        });
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_FUNCTIONAL_HPP
