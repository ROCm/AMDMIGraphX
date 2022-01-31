#ifndef MIGRAPHX_GUARD_KERNELS_VEC_HPP
#define MIGRAPHX_GUARD_KERNELS_VEC_HPP

#include <migraphx/kernels/types.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/functional.hpp>

namespace migraphx {

template <class T, index_int N>
constexpr auto vec_size(vec<T, N>)
{
    return index_constant<N>{};
}

template <class T>
constexpr auto vec_size(T, ...) // NOLINT
{
    return index_constant<0>{};
}

template <class T>
constexpr auto vec_size()
{
    return decltype(vec_size(T{})){};
}

template <class... Ts>
constexpr auto is_any_vec()
{
    if constexpr(sizeof...(Ts) == 0)
        return false_type{};
    else
        return bool_constant<((vec_size<Ts>() + ...) > 0)>{};
}

template <class T, class I>
constexpr auto vec_at(T x, I i)
{
    if constexpr(vec_size<T>() == 0)
        return x;
    else
    {
        MIGRAPHX_ASSERT(i < vec_size<T>());
        return x[i];
    }
}

template <class... Ts>
constexpr auto common_vec_size()
{
    return fold([](auto x, auto y) {
        if constexpr(x > y)
            return x;
        else
            return y;
    })(vec_size<Ts>()...);
}

template <index_int N, class T>
__device__ __host__ auto as_vec(T* x)
{
    if constexpr(N == 0)
        return x;
    else
        return reinterpret_cast<vec<T, N>*>(x);
}

template <class... Ts>
constexpr auto vec_transform(Ts... xs)
{
    return [=](auto f) {
        if constexpr(is_any_vec<Ts...>())
        {
            using type             = decltype(f(vec_at(xs, 0)...));
            constexpr auto size    = common_vec_size<Ts...>();
            vec<type, size> result = {0};
            for(int i = 0; i < size; i++)
                result[i] = f(vec_at(xs, i)...);
            return result;
        }
        else
        {
            return f(xs...);
        }
    };
}

} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_VEC_HPP
