#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_SCATTER_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_SCATTER_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument scatter(
    hipStream_t stream, argument result, argument arg0, argument arg1, argument arg2, int64_t axis);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
