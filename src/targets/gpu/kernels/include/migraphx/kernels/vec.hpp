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

template <class T>
using vec_type = decltype(vec_at(T{}, 0));

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

// Bools can not be used as a vector type so convert it to uint8
template <class T>
__device__ __host__ T* remove_bool(T* x)
{
    return x;
}

inline __device__ __host__ uint8_t* remove_bool(bool* x) { return reinterpret_cast<uint8_t*>(x); }

template <index_int N, class T>
__device__ __host__ auto as_vec(T* x)
{
    if constexpr(N < 2)
        return x;
    else
        return reinterpret_cast<vec<T, N>*>(x);
}

template <class T, index_int N>
using safe_vec = vec<std::conditional_t<std::is_same<T, bool>{}, uint8_t, T>, N>;

template <class... Ts>
constexpr auto vec_transform(Ts... xs)
{
    return [=](auto f) {
        if constexpr(is_any_vec<Ts...>())
        {
            using type                  = decltype(f(vec_at(xs, 0)...));
            constexpr auto size         = common_vec_size<Ts...>();
            safe_vec<type, size> result = {0};
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

// Return a vector type of N from index i in another larger vector
// N will be 2 for half2 packing
template <index_int N, class T, class I>
constexpr vec<vec_type<T>, N> vec_packed_at(T x, I i)
{
    if constexpr(vec_size<T>() == 0)
        return vec<T, N>{x};
    else
    {
        MIGRAPHX_ASSERT((i + N) < vec_size<T>());
        vec<vec_type<T>, N> result = {0};
        for(int j = 0; j < N; j++)
        {
            result[j] = x[i + j];
        }
        return result;
    }
}

template <index_int N, class... Ts>
constexpr auto vec_packed_transform(Ts... xs)
{
    return [=](auto f) {
        if constexpr(is_any_vec<Ts...>())
        {
            using type                  = vec_type<decltype(f(vec_packed_at<N>(xs, 0)...))>;
            constexpr auto size         = common_vec_size<Ts...>();
            safe_vec<type, size> result = {0};
            for(int i = 0; i < size / N; i++)
            {
                // Call the function with packed vectors
                safe_vec<type, N> r = f(vec_packed_at<N>(xs, i * N)...);
                // Copy the packed vectors to the result
                for(int j = 0; j < N; j++)
                    result[i * N + j] = r[j];
            }
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
