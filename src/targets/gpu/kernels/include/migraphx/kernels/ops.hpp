#ifndef MIGRAPHX_GUARD_KERNELS_OPS_HPP
#define MIGRAPHX_GUARD_KERNELS_OPS_HPP

#include <migraphx/kernels/math.hpp>

namespace migraphx {
namespace op {

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
    index_int item_num = 1;
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
        return migraphx::max(x, y);
    }
};

struct min
{
    template <class T, class U>
    MIGRAPHX_DEVICE_CONSTEXPR auto operator()(T x, U y) const
    {
        return migraphx::min(x, y);
    }
};
} // namespace op

struct lowest
{
    template <class T>
    constexpr operator T() const
    {
        return numeric_lowest<T>();
    }
};

struct highest
{
    template <class T>
    constexpr operator T() const
    {
        return numeric_max<T>();
    }
};
} // namespace migraphx
#endif // MIGRAPHX_GUARD_KERNELS_OPS_HPP
