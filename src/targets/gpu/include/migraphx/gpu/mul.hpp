#ifndef MIGRAPHX_GUARD_RTGLIB_MUL_HPP
#define MIGRAPHX_GUARD_RTGLIB_MUL_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/mul.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_mul : binary_device<hip_mul, device::mul>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
