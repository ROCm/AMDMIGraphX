#ifndef MIGRAPHX_GUARD_KERNELS_PRESTORE_HPP
#define MIGRAPHX_GUARD_KERNELS_PRESTORE_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/vec.hpp>

namespace migraphx {

template <class T, class Shape>
constexpr auto make_packed_tensor(T* x, Shape)
{
    constexpr auto s = Shape{};
    if constexpr(s.packed())
    {
        return make_tensor_view(x, s);
    }
    else
    {
        return make_tensor_view(x, make_shape_from_permutation(s.lens, find_permutation(s)));
    }
}

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
            auto r = f(b);
            // TODO: Use vectorize copy if packed
            idx.local_stride(size, [&](auto i) { x[i] = b[i]; });
            return r;
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
