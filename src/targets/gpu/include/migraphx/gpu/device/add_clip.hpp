
#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_ADD_CLIP_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_ADD_CLIP_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void add_clip(hipStream_t stream,
              const argument& result,
              const argument& arg1,
              const argument& arg2,
              float max,
              float min);

void add_clip(hipStream_t stream,
              const argument& result,
              const argument& arg1,
              const argument& arg2,
              const argument& arg3,
              float max,
              float min);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
