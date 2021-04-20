#ifndef MIGRAPHX_GUARD_RTGLIB_LOGICAL_OR_HPP
#define MIGRAPHX_GUARD_RTGLIB_LOGICAL_OR_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/logical_or.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_logical_or : binary_device<hip_logical_or, device::logical_or>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
