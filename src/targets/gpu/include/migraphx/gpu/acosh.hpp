#ifndef MIGRAPHX_GUARD_RTGLIB_ACOSH_HPP
#define MIGRAPHX_GUARD_RTGLIB_ACOSH_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/acosh.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_acosh : unary_device<hip_acosh, device::acosh>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
