#ifndef MIGRAPHX_GUARD_RTGLIB_TANH_HPP
#define MIGRAPHX_GUARD_RTGLIB_TANH_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/tanh.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_tanh : unary_device<hip_tanh, device::tanh>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
