
#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_FP_CONVERSION_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_FP_CONVERSION_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument
fp_conversion(hipStream_t stream, const shape& output_shape, const std::vector<argument>& args);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
