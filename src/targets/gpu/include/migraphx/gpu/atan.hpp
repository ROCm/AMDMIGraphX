#ifndef MIGRAPHX_GUARD_RTGLIB_ATAN_HPP
#define MIGRAPHX_GUARD_RTGLIB_ATAN_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/atan.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_atan : unary_device<hip_atan, device::atan>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
