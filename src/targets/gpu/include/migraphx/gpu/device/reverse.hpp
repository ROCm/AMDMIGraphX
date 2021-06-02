#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_REVERSE_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_REVERSE_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument reverse(hipStream_t stream, argument result, argument arg1, int64_t axis); //TODO: why int64_t for something small?

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
