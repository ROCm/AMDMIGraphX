#ifndef MIGRAPHX_GUARD_RTGLIB_LOG_HPP
#define MIGRAPHX_GUARD_RTGLIB_LOG_HPP

#include <migraphx/gpu/oper.hpp>
#include <migraphx/gpu/device/log.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_log : unary_device<hip_log, device::log>
{
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
