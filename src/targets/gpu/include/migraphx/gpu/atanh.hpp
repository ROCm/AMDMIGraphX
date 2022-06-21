#ifndef MIGRAPHX_GUARD_RTGLIB_ATANH_HPP
#define MIGRAPHX_GUARD_RTGLIB_ATANH_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/atanh.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_atanh : unary_device<hip_atanh, device::atanh>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
