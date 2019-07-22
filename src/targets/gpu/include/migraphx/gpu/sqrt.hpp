#ifndef MIGRAPHX_GUARD_RTGLIB_SQRT_HPP
#define MIGRAPHX_GUARD_RTGLIB_SQRT_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/sqrt.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_sqrt : unary_device<hip_sqrt, device::sqrt>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
