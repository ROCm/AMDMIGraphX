#ifndef MIGRAPHX_GUARD_RTGLIB_EXP_HPP
#define MIGRAPHX_GUARD_RTGLIB_EXP_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/exp.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_exp : unary_device<hip_exp, device::exp>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
