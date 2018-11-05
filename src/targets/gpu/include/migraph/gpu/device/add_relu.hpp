
#ifndef MIGRAPH_GUARD_RTGLIB_DEVICE_ADD_RELU_HPP
#define MIGRAPH_GUARD_RTGLIB_DEVICE_ADD_RELU_HPP

#include <migraph/argument.hpp>
#include <migraph/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraph { inline namespace MIGRAPH_INLINE_NS {
namespace gpu {
namespace device {

void add_relu(hipStream_t stream,
              const argument& result,
              const argument& arg1,
              const argument& arg2);

void add_relu(hipStream_t stream,
              const argument& result,
              const argument& arg1,
              const argument& arg2,
              const argument& arg3);

} // namespace device
} // namespace gpu
} // inline namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
