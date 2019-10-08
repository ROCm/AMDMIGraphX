#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_FAST_DIV_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_FAST_DIV_HPP

#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

constexpr const std::size_t fast_div_shift = 42;
std::size_t encode_divisor(std::size_t divisor)
{
    if(divisor == 0)
        return 0;
    auto p = std::size_t{1} << fast_div_shift;
    // std::cout << divisor << " -> " << ((p + divisor - 1) / divisor) << std::endl;
    return (p + divisor - 1) / divisor;
}

inline constexpr bool is_divisor_encodable(std::size_t i)
{
    return i < (std::size_t{1} << (fast_div_shift / 2));
}

MIGRAPHX_DEVICE_CONSTEXPR std::size_t fast_div(std::size_t dividend, std::size_t encoded_divisor)
{
    return (dividend * encoded_divisor) >> fast_div_shift;
}

MIGRAPHX_DEVICE_CONSTEXPR std::size_t
remainder(std::size_t result, std::size_t dividend, std::size_t divisor)
{
    return dividend - divisor * result;
}

MIGRAPHX_DEVICE_CONSTEXPR std::size_t
fast_mod(std::size_t dividend, std::size_t divisor, std::size_t encoded_divisor)
{
    return remainder(fast_div(dividend, encoded_divisor), dividend, divisor);
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
