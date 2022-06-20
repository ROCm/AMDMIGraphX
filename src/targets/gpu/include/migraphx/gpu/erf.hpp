#ifndef MIGRAPHX_GUARD_RTGLIB_ERF_HPP
#define MIGRAPHX_GUARD_RTGLIB_ERF_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/erf.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_erf : unary_device<hip_erf, device::erf>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
