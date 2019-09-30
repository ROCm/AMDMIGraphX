#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_FAST_DIV_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_FAST_DIV_HPP

#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

constexpr const std::size_t fast_div_shift = 31;
MIGRAPHX_DEVICE_CONSTEXPR std::size_t encode_divisor(std::size_t divisor)
{
    if(divisor == 0)
        return 0;
    return (1L << fast_div_shift) / divisor + 1;
}

MIGRAPHX_DEVICE_CONSTEXPR std::size_t fast_div(std::size_t dividend, std::size_t encoded_divisor)
{
    return (dividend * encoded_divisor) >> fast_div_shift;
}

MIGRAPHX_DEVICE_CONSTEXPR std::size_t fast_mod(std::size_t dividend, std::size_t divisor, std::size_t encoded_divisor)
{
    return dividend - divisor * fast_div(dividend, encoded_divisor);
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
