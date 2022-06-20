#ifndef MIGRAPHX_GUARD_RTGLIB_SIGMOID_HPP
#define MIGRAPHX_GUARD_RTGLIB_SIGMOID_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/sigmoid.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_sigmoid : unary_device<hip_sigmoid, device::sigmoid>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
