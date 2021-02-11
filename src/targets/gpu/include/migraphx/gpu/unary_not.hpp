#ifndef MIGRAPHX_GUARD_RTGLIB_UNARY_NOT_HPP
#define MIGRAPHX_GUARD_RTGLIB_UNARY_NOT_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/unary_not.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_unary_not : binary_device<hip_unary_not, device::unary_not>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
