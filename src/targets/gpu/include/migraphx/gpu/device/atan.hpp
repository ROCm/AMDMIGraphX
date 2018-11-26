#ifndef MIGRAPH_GUARD_RTGLIB_DEVICE_ATAN_HPP
#define MIGRAPH_GUARD_RTGLIB_DEVICE_ATAN_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {
namespace gpu {
namespace device {

void atan(hipStream_t stream, const argument& result, const argument& arg);

} // namespace device
} // namespace gpu
} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx

#endif
