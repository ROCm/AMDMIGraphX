#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_BASIC_OPS_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_BASIC_OPS_HPP

#include <migraphx/kernels/types.hpp>

namespace migraphx {

struct sum
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return x + y;
    }
};

struct product
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return x * y;
    }
};

struct id
{
    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x) const
    {
        return x;
    }
};

struct mean
{
    size_t item_num = 1;
    template <class T>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x) const
    {
        return x / static_cast<T>(item_num);
    }
};

struct max
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return (x > y) ? x : y;
    }
};

struct min
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return (x < y) ? x : y;
    }
};

struct lowest
{
    template <class T>
    constexpr __device__ __host__ operator T() const
    {
        // return device_cast(std::numeric_limits<host_type<T>>::lowest());
        return std::numeric_limits<T>::lowest();
    }
};

struct highest
{
    template <class T>
    constexpr __device__ __host__ operator T() const
    {
        // return device_cast(std::numeric_limits<host_type<T>>::max());
        return std::numeric_limits<T>::max();
    }
};

} // namespace migraphx
#endif // MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_BASIC_OPS_HPP
