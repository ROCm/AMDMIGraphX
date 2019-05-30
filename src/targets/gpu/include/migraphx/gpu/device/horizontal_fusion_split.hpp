#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_SPLIT_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_SPLIT_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument split(hipStream_t stream,
               const migraphx::shape& output_shape,
               std::vector<argument> args,
               unsigned offset);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
