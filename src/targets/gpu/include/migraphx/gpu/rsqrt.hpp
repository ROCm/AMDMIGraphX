#ifndef MIGRAPHX_GUARD_RTGLIB_RSQRT_HPP
#define MIGRAPHX_GUARD_RTGLIB_RSQRT_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/rsqrt.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_rsqrt : unary_device<hip_rsqrt, device::rsqrt>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
