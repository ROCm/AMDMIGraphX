#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_NONZERO_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_NONZERO_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument nonzero(hipStream_t stream, argument result, argument arg);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
