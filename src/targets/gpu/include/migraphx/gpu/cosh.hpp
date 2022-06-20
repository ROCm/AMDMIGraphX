#ifndef MIGRAPHX_GUARD_RTGLIB_COSH_HPP
#define MIGRAPHX_GUARD_RTGLIB_COSH_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/cosh.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_cosh : unary_device<hip_cosh, device::cosh>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
