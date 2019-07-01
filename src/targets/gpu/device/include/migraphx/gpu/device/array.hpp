
#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_ARRAY_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_ARRAY_HPP

#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

template <class T, std::size_t N>
struct hip_array
{
    T d[N];
    MIGRAPHX_DEVICE_CONSTEXPR T& operator[](std::size_t i) { return d[i]; }
    MIGRAPHX_DEVICE_CONSTEXPR const T& operator[](std::size_t i) const { return d[i]; }

    MIGRAPHX_DEVICE_CONSTEXPR T* data() { return d; }
    MIGRAPHX_DEVICE_CONSTEXPR const T* data() const { return d; }

    MIGRAPHX_DEVICE_CONSTEXPR std::integral_constant<std::size_t, N> size() const { return {}; }

    MIGRAPHX_DEVICE_CONSTEXPR T* begin() { return d; }
    MIGRAPHX_DEVICE_CONSTEXPR const T* begin() const { return d; }

    MIGRAPHX_DEVICE_CONSTEXPR T* end() { return d + size(); }
    MIGRAPHX_DEVICE_CONSTEXPR const T* end() const { return d + size(); }

    MIGRAPHX_DEVICE_CONSTEXPR T dot(const hip_array& x) const
    {
        T result = 0;
        for(std::size_t i = 0; i < N; i++)
            result += x[i] * d[i];
        return result;
    }

    MIGRAPHX_DEVICE_CONSTEXPR T product() const
    {
        T result = 1;
        for(std::size_t i = 0; i < N; i++)
            result *= d[i];
        return result;
    }

    friend MIGRAPHX_DEVICE_CONSTEXPR hip_array operator*(const hip_array& x, const hip_array& y)
    {
        hip_array result;
        for(std::size_t i = 0; i < N; i++)
            result[i] = x[i] * y[i];
        return result;
    }

    friend MIGRAPHX_DEVICE_CONSTEXPR hip_array operator+(const hip_array& x, const hip_array& y)
    {
        hip_array result{};
        for(std::size_t i = 0; i < N; i++)
            result[i] = x[i] + y[i];
        return result;
    }
};

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
