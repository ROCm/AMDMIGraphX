#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_GATHERND_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_GATHERND_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument
gathernd(hipStream_t stream, argument result, argument arg0, argument arg1, const int& batch_dims);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
