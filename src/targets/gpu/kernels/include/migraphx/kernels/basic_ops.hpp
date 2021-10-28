#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_BASIC_OPS_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_BASIC_OPS_HPP

#include <migraphx/kernels/types.hpp>

namespace migraphx {

struct sum
{
    template <class T, class U>
    constexpr auto operator()(T x, U y) const
    {
        return x + y;
    }
};

struct product
{
    template <class T, class U>
    constexpr auto operator()(T x, U y) const
    {
        return x * y;
    }
};

struct id
{
    template <class T>
    constexpr auto operator()(T x) const
    {
        return x;
    }
};

struct mean
{
    size_t item_num = 1;
    template <class T>
    constexpr auto operator()(T x) const
    {
        return x / static_cast<T>(item_num);
    }
};

struct max_f
{
    template <class T, class U>
    constexpr auto operator()(T x, U y) const
    {
        return (x > y) ? x : y;
    }
};
inline constexpr auto max = max_f{};

struct min_f
{
    template <class T, class U>
    constexpr auto operator()(T x, U y) const
    {
        return (x < y) ? x : y;
    }
};
inline constexpr auto min = min_f{};

struct lowest
{
    template <class T>
    constexpr operator T() const
    {
        return std::numeric_limits<T>::lowest();
    }
};

struct highest
{
    template <class T>
    constexpr operator T() const
    {
        return std::numeric_limits<T>::max();
    }
};

} // namespace migraphx
#endif // MIGRAPHX_GUARD_AMDMIGRAPHX_KERNELS_BASIC_OPS_HPP
