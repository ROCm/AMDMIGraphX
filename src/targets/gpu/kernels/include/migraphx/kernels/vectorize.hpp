#ifndef MIGRAPHX_GUARD_KERNELS_VECTORIZE_HPP
#define MIGRAPHX_GUARD_KERNELS_VECTORIZE_HPP

#include <migraphx/kernels/tensor_view.hpp>

namespace migraphx {

template <class T, index_int N>
constexpr auto vec_size(vec<T, N>)
{
    return index_constant<N>{};
}

template <class T>
constexpr auto vec_size(T, ...)
{
    return index_constant<0>{};
}

template <class T>
constexpr auto vec_size()
{
    return decltype(vec_size(T{})){};
}

template <class T>
constexpr auto tensor_vec_size(T)
{
    return vec_size<typename T::type>();
}

template <index_int N, class Shape>
constexpr auto as_vec_shape(Shape s)
{
    auto lens = transform(s.lens, s.strides, [](auto len, auto stride) {
        if(stride == 1)
            return len / N;
        else
            return len;
    });
    return make_shape(lens, s.strides);
}

template <index_int N, class T>
__device__ __host__ auto as_vec(T* x)
{
    if constexpr(N == 0)
        return x;
    else
        return reinterpret_cast<vec<T, N>*>(x);
}

template <index_int N, class T>
__device__ __host__ auto as_vec(T x)
{
    if constexpr(N == 0)
        return x;
    else
        return make_tensor_view(as_vec<N>(x.data()), as_vec_shape<N>(x.get_shape()));
}

template <class IntegralConstant, class T>
__device__ __host__ auto as_vec(IntegralConstant ic, T&& x)
{
    return as_vec<ic>(x);
}

template <index_int N, class Shape>
constexpr bool is_vectorizable(Shape s)
{
    auto it = find(s.strides.begin(), s.strides.end(), 1);
    if(it == s.strides.end())
        return false;
    index_int i = it - s.strides.begin();
    return (s.lens[i] % N) == 0;
}

template <index_int N, class... Shapes>
constexpr auto is_vectorizable()
{
    return bool_constant<(is_vectorizable<N>(Shapes{}) and ...)>{};
}

template <class T>
__device__ __host__ auto vectorize(T x)
{
    if constexpr(is_vectorizable<4, decltype(x.get_shape())>())
        return as_vec<4>(x);
    else
        return x;
}

template <class... Ts>
__device__ __host__ auto auto_vectorize(Ts... xs)
{
    return [=](auto f) {
        constexpr bool packed = (decltype(xs.get_shape()){}.packed() and ...);
        if constexpr(packed)
            f(vectorize(xs)...);
        else
            f(xs...);
    };
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_VECTORIZE_HPP
