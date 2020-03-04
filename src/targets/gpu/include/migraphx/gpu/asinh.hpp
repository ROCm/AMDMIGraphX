#ifndef MIGRAPHX_GUARD_RTGLIB_ASINH_HPP
#define MIGRAPHX_GUARD_RTGLIB_ASINH_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/asinh.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_asinh : unary_device<hip_asinh, device::asinh>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
