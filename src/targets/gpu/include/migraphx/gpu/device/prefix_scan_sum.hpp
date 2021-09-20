#ifndef MIGRAPHX_GUARD_DEVICE_PREFIX_SCAN_SUM_HPP
#define MIGRAPHX_GUARD_DEVICE_PREFIX_SCAN_SUM_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void prefix_scan_sum(hipStream_t stream, const argument& result, const argument& arg, int32_t axis);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_DEVICE_PREFIX_SCAN_SUM_HPP
