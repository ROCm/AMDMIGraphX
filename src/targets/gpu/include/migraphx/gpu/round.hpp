#ifndef MIGRAPHX_GUARD_RTGLIB_ROUND_HPP
#define MIGRAPHX_GUARD_RTGLIB_ROUND_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/round.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_round : unary_device<hip_round, device::round>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
