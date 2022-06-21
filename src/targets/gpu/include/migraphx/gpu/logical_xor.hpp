#ifndef MIGRAPHX_GUARD_RTGLIB_LOGICAL_XOR_HPP
#define MIGRAPHX_GUARD_RTGLIB_LOGICAL_XOR_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/logical_xor.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_logical_xor : binary_device<hip_logical_xor, device::logical_xor>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
