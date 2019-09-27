#ifndef MIGRAPHX_GUARD_RTGLIB_FLOOR_HPP
#define MIGRAPHX_GUARD_RTGLIB_FLOOR_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/floor.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_floor : unary_device<hip_floor, device::floor>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
