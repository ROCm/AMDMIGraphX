#ifndef MIGRAPHX_GUARD_RTGLIB_WHERE_HPP
#define MIGRAPHX_GUARD_RTGLIB_WHERE_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/where.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_where : ternary_device<hip_where, device::where>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
