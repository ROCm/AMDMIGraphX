#ifndef MIGRAPHX_GUARD_RTGLIB_MAX_HPP
#define MIGRAPHX_GUARD_RTGLIB_MAX_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/max.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_max : binary_device<hip_max, device::max>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
