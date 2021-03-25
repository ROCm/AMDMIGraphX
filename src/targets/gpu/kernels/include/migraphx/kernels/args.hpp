#ifndef MIGRAPHX_GUARD_KERNELS_ARGS_HPP
#define MIGRAPHX_GUARD_KERNELS_ARGS_HPP

#include <migraphx/kernels/types.hpp>

namespace migraphx {

template <std::size_t N>
struct arg
{
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

// Use template specialization since ADL is broken on hcc
template <std::size_t>
struct make_tensor;

template <class F, std::size_t... Ns, class... Ts>
__device__ auto make_tensors_impl(F f, seq<Ns...>, Ts*... xs)
{
    f(make_tensor<Ns>::apply(xs)...);
}

template <class... Ts>
__device__ auto make_tensors(Ts*... xs)
{
    return [=](auto f) { make_tensors_impl(f, gens<sizeof...(Ts)>{}, xs...); };
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_ARGS_HPP