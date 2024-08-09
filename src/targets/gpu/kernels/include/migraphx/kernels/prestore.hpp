#ifndef MIGRAPHX_GUARD_KERNELS_PRESTORE_HPP
#define MIGRAPHX_GUARD_KERNELS_PRESTORE_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/copy.hpp>

namespace migraphx {

template <bool B, class T>
__device__ auto prestore_copy(index idx, T x)
{
    return [=](auto f) {
        if constexpr(B)
        {
            using type          = typename T::type;
            constexpr auto size = get_shape_c<T>{}.elements();
            __shared__ type buffer[size];
            auto b = make_packed_tensor(buffer, get_shape_c<T>{});
            f(b);
            local_tensor_copy(idx, b, x);
        }
        else
        {
            return f(x);
        }
    };
}

template <bool... Bs>
__device__ auto auto_prestore(index idx)
{
    return make_transform([=](auto f, auto... xs) {
        static_assert(sizeof...(Bs) == sizeof...(xs));
        auto invoke = [=](auto... ys) {
            f(ys...);
            if constexpr((Bs or ...))
                __syncthreads();
        };
        join(invoke, prestore_copy<Bs>(idx, xs)...);
    });
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_PRESTORE_HPP
