#ifndef MIGRAPHX_GUARD_KERNELS_POINTWISE_HPP
#define MIGRAPHX_GUARD_KERNELS_POINTWISE_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/math.hpp>
#include <migraphx/kernels/preload.hpp>
#include <migraphx/kernels/vectorize.hpp>
#include <migraphx/kernels/args.hpp>

namespace migraphx {

template <class T>
struct implicit_conversion_op
{
    T x;

    template <index_int N, class U>
    constexpr operator vec<U, N>() const
    {
        static_assert(vec_size<T>() == N, "Vector mismatch size");
        return __builtin_convertvector(x, vec<U, N>);
    }

    template <class U>
    constexpr operator U() const
    {
        return x;
    }
};

template <class T>
constexpr implicit_conversion_op<T> implicit_conversion(T x)
{
    return {x};
}

template <class F, class T, class... Ts>
__device__ void pointwise_tensor(index idx, F f, T out, Ts... xs)
{
    preload<typename T::type>(idx, xs...)([&](auto... ps) {
        idx.global_stride(out.get_shape().elements(),
                          [&](auto i) { out[i] = implicit_conversion(f(ps[i]...)); });
    });
}

template <class F, class... Ts>
__device__ void pointwise(F f, Ts*... ps)
{
    auto t = transform_args(make_tensors(), rotate_last(), auto_vectorize());
    t(ps...)([&](auto... xs) {
        auto idx = make_index();
        pointwise_tensor(idx, f, xs...);
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_POINTWISE_HPP
