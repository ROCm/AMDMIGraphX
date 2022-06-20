#ifndef MIGRAPHX_GUARD_RTGLIB_ACOS_HPP
#define MIGRAPHX_GUARD_RTGLIB_ACOS_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/acos.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_acos : unary_device<hip_acos, device::acos>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
