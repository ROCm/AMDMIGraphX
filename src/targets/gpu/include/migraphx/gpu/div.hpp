#ifndef MIGRAPHX_GUARD_RTGLIB_DIV_HPP
#define MIGRAPHX_GUARD_RTGLIB_DIV_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/div.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_div : binary_device<hip_div, device::div>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
