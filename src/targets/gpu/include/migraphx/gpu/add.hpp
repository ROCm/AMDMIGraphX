#ifndef MIGRAPHX_GUARD_RTGLIB_ADD_HPP
#define MIGRAPHX_GUARD_RTGLIB_ADD_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/add.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_add : binary_device<hip_add, device::add>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
