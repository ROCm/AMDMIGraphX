#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_GATHER_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_GATHER_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument gather(hipStream_t stream, argument result, argument arg1, argument arg2, int axis);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
