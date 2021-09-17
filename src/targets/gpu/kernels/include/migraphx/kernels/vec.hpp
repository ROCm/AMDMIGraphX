#ifndef MIGRAPHX_GUARD_KERNELS_VEC_HPP
#define MIGRAPHX_GUARD_KERNELS_VEC_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/integral_constant.hpp>

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

template <index_int N, class T>
__device__ __host__ auto as_vec(T* x)
{
    if constexpr(N == 0)
        return x;
    else
        return reinterpret_cast<vec<T, N>*>(x);
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_VEC_HPP
