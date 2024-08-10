#ifndef MIGRAPHX_GUARD_KERNELS_PRESTORE_HPP
#define MIGRAPHX_GUARD_KERNELS_PRESTORE_HPP

#include <migraphx/kernels/index.hpp>
#include <migraphx/kernels/functional.hpp>
#include <migraphx/kernels/tensor_view.hpp>
#include <migraphx/kernels/copy.hpp>

namespace migraphx {

template<class Shape>
constexpr auto tile_pad_shape(Shape)
{
    constexpr Shape s{};
    constexpr auto axis = s.strides.size() - _c<1>;
    constexpr auto strides = transform_i(s.strides, [](auto stride, auto i) {
        if constexpr(i == decltype(axis){})
        {
            if(stride < 64)
                return index_int{65};
            return stride+1;
        }
        else
        {
            return stride;
        }
    });
    return make_shape(s.lens, strides);
}

template <bool B, class T>
__device__ auto prestore_copy(index idx, T x)
{
    return [=](auto f) {
        if constexpr(B)
        {
            using type          = typename T::type;
            constexpr auto s = tile_pad_shape(make_packed_shape(get_shape_c<T>{}));
            // constexpr auto s = make_packed_shape(get_shape_c<T>{});
            constexpr auto size = s.element_space();
            // println_once("size", size);
            // println_once("s", s);
            // println_once("shape", get_shape_c<T>{});
            __shared__ type buffer[size];
            auto b = make_tensor_view(buffer, s);
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
