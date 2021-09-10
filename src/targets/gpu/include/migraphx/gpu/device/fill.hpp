#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_FILL_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_FILL_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void fill(hipStream_t stream, const argument& result, unsigned long val);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
