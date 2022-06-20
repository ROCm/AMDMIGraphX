#ifndef MIGRAPHX_GUARD_RTGLIB_EQUAL_HPP
#define MIGRAPHX_GUARD_RTGLIB_EQUAL_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/equal.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_equal : binary_device<hip_equal, device::equal>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
