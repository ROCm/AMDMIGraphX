#ifndef MIGRAPHX_GUARD_RTGLIB_COS_HPP
#define MIGRAPHX_GUARD_RTGLIB_COS_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/cos.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_cos : unary_device<hip_cos, device::cos>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
