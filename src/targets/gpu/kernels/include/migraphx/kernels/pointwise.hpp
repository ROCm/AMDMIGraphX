#ifndef MIGRAPHX_GUARD_KERNELS_POINTWISE_HPP
#define MIGRAPHX_GUARD_KERNELS_POINTWISE_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/preload.hpp>
#include <migraphx/kernels/vectorize.hpp>
#include <migraphx/kernels/args.hpp>

namespace migraphx {

template <class F, class T, class... Ts>
__device__ void pointwise_tensor(index idx, F f, T out, Ts... xs)
{
    preload<typename T::type>(idx, xs...)([&](auto... ps) {
        idx.global_stride(out.get_shape().elements(), [&](auto i) {
            auto multi_idx = out.get_shape().multi(i);
            out[multi_idx] = f(ps[multi_idx]...);
        });
    });
}

template <class F, class... Ts>
__device__ void pointwise(F f, Ts*... ps)
{
    auto t = transform_args(make_tensors(), rotate_last());
    t(ps...)([&](auto... xs) {
        auto idx = make_index();
        pointwise_tensor(idx, f, xs...);
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_POINTWISE_HPP
