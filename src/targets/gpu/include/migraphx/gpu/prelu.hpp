#ifndef MIGRAPHX_GUARD_RTGLIB_PRELU_HPP
#define MIGRAPHX_GUARD_RTGLIB_PRELU_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/prelu.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_prelu : binary_device<hip_prelu, device::prelu>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
