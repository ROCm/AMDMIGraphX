#ifndef MIGRAPH_GUARD_MIGRAPHLIB_KERNELS_HPP
#define MIGRAPH_GUARD_MIGRAPHLIB_KERNELS_HPP

#include <migraph/argument.hpp>
#include <migraph/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraph { inline namespace MIGRAPH_INLINE_NS {
namespace gpu {
namespace device {

void contiguous(hipStream_t stream, argument result, argument arg);

} // namespace device
} // namespace gpu
} // inline namespace MIGRAPH_INLINE_NS
} // namespace migraph

#endif
