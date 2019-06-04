#ifndef MIGRAPHX_GUARD_RTGLIB_MIN_HPP
#define MIGRAPHX_GUARD_RTGLIB_MIN_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/min.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_min : binary_device<hip_min, device::min>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
