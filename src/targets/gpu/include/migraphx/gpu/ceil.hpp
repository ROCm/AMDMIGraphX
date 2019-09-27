#ifndef MIGRAPHX_GUARD_RTGLIB_CEIL_HPP
#define MIGRAPHX_GUARD_RTGLIB_CEIL_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/ceil.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_ceil : unary_device<hip_ceil, device::ceil>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
