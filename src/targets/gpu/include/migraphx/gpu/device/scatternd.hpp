#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_SCATTERND_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_SCATTERND_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument
scatternd(hipStream_t stream, argument result, argument arg0, argument arg1, argument arg2);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif