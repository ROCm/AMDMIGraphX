#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_ROIALIGN_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_ROIALIGN_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument roialign(hipStream_t stream,
                  argument result,
                  const std::vector<argument>& args,
                  const std::string& coord_trans_mode,
                  const std::string& pooling_mode,
                  int64_t sampling_ratio,
                  float spatial_scale);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
