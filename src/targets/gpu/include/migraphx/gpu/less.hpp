#ifndef MIGRAPHX_GUARD_RTGLIB_LESS_HPP
#define MIGRAPHX_GUARD_RTGLIB_LESS_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/less.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_less : binary_device<hip_less, device::less>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
