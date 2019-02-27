#ifndef MIGRAPHX_GUARD_RTGLIB_SIN_HPP
#define MIGRAPHX_GUARD_RTGLIB_SIN_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/sin.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_sin : unary_device<hip_sin, device::sin>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
