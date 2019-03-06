#ifndef MIGRAPHX_GUARD_RTGLIB_TAN_HPP
#define MIGRAPHX_GUARD_RTGLIB_TAN_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/tan.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_tan : unary_device<hip_tan, device::tan>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
