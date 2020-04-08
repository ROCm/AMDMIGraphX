#ifndef MIGRAPHX_GUARD_RTGLIB_RECIP_HPP
#define MIGRAPHX_GUARD_RTGLIB_RECIP_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/recip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_recip : unary_device<hip_recip, device::recip>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
