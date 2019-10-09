
#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_ARRAY_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_ARRAY_HPP

#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

// NOLINTNEXTLINE
#define MIGRAPHX_DEVICE_ARRAY_OP(op, binary_op)                                                    \
    MIGRAPHX_DEVICE_CONSTEXPR hip_array& operator op(const hip_array& x)                           \
    {                                                                                              \
        for(std::size_t i = 0; i < N; i++)                                                         \
            d[i] op x[i];                                                                          \
        return *this;                                                                              \
    }                                                                                              \
    MIGRAPHX_DEVICE_CONSTEXPR hip_array& operator op(const T& x)                                   \
    {                                                                                              \
        for(std::size_t i = 0; i < N; i++)                                                         \
            d[i] op x;                                                                             \
        return *this;                                                                              \
    }                                                                                              \
    friend MIGRAPHX_DEVICE_CONSTEXPR hip_array operator binary_op(hip_array x, const hip_array& y) \
    {                                                                                              \
        return x op y;                                                                             \
    }                                                                                              \
    friend MIGRAPHX_DEVICE_CONSTEXPR hip_array operator binary_op(hip_array x, const T& y)         \
    {                                                                                              \
        return x op y;                                                                             \
    }                                                                                              \
    friend MIGRAPHX_DEVICE_CONSTEXPR hip_array operator binary_op(const T& y, hip_array x)         \
    {                                                                                              \
        return x op y;                                                                             \
    }

template <class T, std::size_t N>
struct hip_array
{
    T d[N];
    MIGRAPHX_DEVICE_CONSTEXPR T& operator[](std::size_t i) { return d[i]; }
    MIGRAPHX_DEVICE_CONSTEXPR const T& operator[](std::size_t i) const { return d[i]; }

    MIGRAPHX_DEVICE_CONSTEXPR T& front() { return d[0]; }
    MIGRAPHX_DEVICE_CONSTEXPR const T& front() const { return d[0]; }

    MIGRAPHX_DEVICE_CONSTEXPR T& back() { return d[N - 1]; }
    MIGRAPHX_DEVICE_CONSTEXPR const T& back() const { return d[N - 1]; }

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

    MIGRAPHX_DEVICE_CONSTEXPR T single(std::size_t width = 100) const
    {
        T result = 0;
        T a      = 1;
        for(std::size_t i = 0; i < N; i++)
        {
            result += d[N - i - 1] * a;
            a *= width;
        }
        return result;
    }

    MIGRAPHX_DEVICE_ARRAY_OP(+=, +)
    MIGRAPHX_DEVICE_ARRAY_OP(*=, *)
    MIGRAPHX_DEVICE_ARRAY_OP(/=, /)
    MIGRAPHX_DEVICE_ARRAY_OP(%=, %)
    MIGRAPHX_DEVICE_ARRAY_OP(&=, &)
    MIGRAPHX_DEVICE_ARRAY_OP(|=, |)
    MIGRAPHX_DEVICE_ARRAY_OP(^=, ^)

    friend MIGRAPHX_DEVICE_CONSTEXPR bool operator==(const hip_array& x, const hip_array& y)
    {
        for(std::size_t i = 0; i < N; i++)
        {
            if(x[i] != y[i])
                return false;
        }
        return true;
    }

    friend MIGRAPHX_DEVICE_CONSTEXPR bool operator!=(const hip_array& x, const hip_array& y)
    {
        return !(x == y);
    }
    // This uses the product order rather than lexical order
    friend MIGRAPHX_DEVICE_CONSTEXPR bool operator<(const hip_array& x, const hip_array& y)
    {
        for(std::size_t i = 0; i < N; i++)
        {
            if(not(x[i] < y[i]))
                return false;
        }
        return true;
    }
    friend MIGRAPHX_DEVICE_CONSTEXPR bool operator>(const hip_array& x, const hip_array& y)
    {
        return y < x;
    }
    friend MIGRAPHX_DEVICE_CONSTEXPR bool operator<=(const hip_array& x, const hip_array& y)
    {
        return (x < y) or (x == y);
    }
    friend MIGRAPHX_DEVICE_CONSTEXPR bool operator>=(const hip_array& x, const hip_array& y)
    {
        return (y < x) or (x == y);
    }

    MIGRAPHX_DEVICE_CONSTEXPR hip_array carry(hip_array result) const
    {
        uint32_t overflow = 0;
        for(std::ptrdiff_t i = result.size() - 1; i > 0; i--)
        {
            auto z = result[i] + overflow;
            // Reset overflow
            overflow = 0;
            // Compute overflow using while loop instead of mod
            while(z >= d[i])
            {
                z -= d[i];
                overflow += 1;
            }
            result[i] = z;
        }
        result[0] += overflow;
        return result;
    }
};

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
