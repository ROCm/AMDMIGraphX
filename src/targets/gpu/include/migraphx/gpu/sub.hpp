#ifndef MIGRAPHX_GUARD_RTGLIB_SUB_HPP
#define MIGRAPHX_GUARD_RTGLIB_SUB_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/sub.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_sub : binary_device<hip_sub, device::sub>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
