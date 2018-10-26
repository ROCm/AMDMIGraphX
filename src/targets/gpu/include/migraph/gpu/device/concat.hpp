#ifndef MIGRAPH_GUARD_RTGLIB_DEVICE_CONCAT_HPP
#define MIGRAPH_GUARD_RTGLIB_DEVICE_CONCAT_HPP

#include <migraph/argument.hpp>
#include <hip/hip_runtime_api.h>

namespace migraph {
namespace gpu {
namespace device {

argument
concat(hipStream_t stream, const shape& output_shape, std::vector<argument> args, std::vector<std::size_t> offsets);

} // namespace device
} // namespace gpu
} // namespace migraph

#endif
