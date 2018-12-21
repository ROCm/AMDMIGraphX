#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_SHAPE_OF_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_SHAPE_OF_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void shape_of(hipStream_t stream, const argument& result, const argument& ins);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
