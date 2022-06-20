#ifndef MIGRAPHX_GUARD_DEVICE_REDUCE_OPS_HPP
#define MIGRAPHX_GUARD_DEVICE_REDUCE_OPS_HPP

#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

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
    __device__ __host__ operator T() const
    {
        return device_cast(std::numeric_limits<host_type<T>>::lowest());
    }
};

struct highest
{
    template <class T>
    __device__ __host__ operator T() const
    {
        return device_cast(std::numeric_limits<host_type<T>>::max());
    }
};

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_DEVICE_REDUCE_OPS_HPP
