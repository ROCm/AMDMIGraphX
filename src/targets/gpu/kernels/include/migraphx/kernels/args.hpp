#ifndef MIGRAPHX_GUARD_KERNELS_ARGS_HPP
#define MIGRAPHX_GUARD_KERNELS_ARGS_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/functional.hpp>

namespace migraphx {

// Use template specialization since ADL is broken on hcc
template <index_int>
struct make_tensor;

template <class F, index_int... Ns, class... Ts>
__device__ auto make_tensors_impl(F f, detail::seq<Ns...>, Ts*... xs)
{
    return f(make_tensor<Ns>::apply(xs)...);
}

inline __device__ auto make_tensors()
{
    return [](auto*... xs) {
        return [=](auto f) { return make_tensors_impl(f, detail::gens<sizeof...(xs)>{}, xs...); };
    };
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_ARGS_HPP
