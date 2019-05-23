#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_SOFTMAX_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_SOFTMAX_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument softmax(hipStream_t stream,
                    const migraphx::shape& output_shape,
                    std::vector<migraphx::argument> args,
                    int axis);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
