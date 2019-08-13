#ifndef MIGRAPHX_GUARD_RTGLIB_RELU_HPP
#define MIGRAPHX_GUARD_RTGLIB_RELU_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/relu.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_relu : unary_device<hip_relu, device::relu>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
