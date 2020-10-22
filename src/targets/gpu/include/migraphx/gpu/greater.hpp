#ifndef MIGRAPHX_GUARD_RTGLIB_GREATER_HPP
#define MIGRAPHX_GUARD_RTGLIB_GREATER_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/greater.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_greater : binary_device<hip_greater, device::greater>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
