#ifndef MIGRAPHX_GUARD_RTGLIB_SINH_HPP
#define MIGRAPHX_GUARD_RTGLIB_SINH_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/sinh.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_sinh : unary_device<hip_sinh, device::sinh>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
