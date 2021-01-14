#ifndef MIGRAPHX_GUARD_RTGLIB_ANDD_HPP
#define MIGRAPHX_GUARD_RTGLIB_ANDD_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/andd.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_andd : binary_device<hip_andd, device::andd>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
