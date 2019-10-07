#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_FAST_DIV_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_FAST_DIV_HPP

#include <migraphx/gpu/device/types.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

constexpr const uint64_t fast_div_shift = 42;
MIGRAPHX_DEVICE_CONSTEXPR uint64_t encode_divisor(uint64_t divisor)
{
    if(divisor == 0)
        return 0;
    auto p = uint64_t{1} << fast_div_shift;
    return (p + divisor - 1) / divisor;
}

inline constexpr bool is_divisor_encodable(uint64_t i)
{
    return i < uint64_t{1} << (fast_div_shift / 2);
}

MIGRAPHX_DEVICE_CONSTEXPR uint64_t fast_div(uint64_t dividend, uint64_t encoded_divisor)
{
    return (dividend * encoded_divisor) >> fast_div_shift;
}

MIGRAPHX_DEVICE_CONSTEXPR uint64_t
remainder(uint64_t result, uint64_t dividend, uint64_t divisor)
{
    return dividend - divisor * result;
}

MIGRAPHX_DEVICE_CONSTEXPR uint64_t
fast_mod(uint64_t dividend, uint64_t divisor, uint64_t encoded_divisor)
{
    return remainder(fast_div(dividend, encoded_divisor), dividend, divisor);
}

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
