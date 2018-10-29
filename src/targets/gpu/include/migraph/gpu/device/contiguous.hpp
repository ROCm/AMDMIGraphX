#ifndef MIGRAPH_GUARD_MIGRAPHLIB_KERNELS_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_KERNELS_HPP

#include <migraph/argument.hpp>
#include <hip/hip_runtime_api.h>

namespace migraph {
namespace gpu {
namespace device {

void contiguous(hipStream_t stream, argument result, argument arg);

} // namespace device
} // namespace gpu
} // namespace migraph

#endif
