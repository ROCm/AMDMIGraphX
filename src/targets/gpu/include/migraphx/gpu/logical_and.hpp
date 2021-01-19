#ifndef MIGRAPHX_GUARD_RTGLIB_LOGICLA_AND_HPP
#define MIGRAPHX_GUARD_RTGLIB_LOGICLA_AND_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/logical_and.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_logical_and : binary_device<hip_logical_and, device::logical_and>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
