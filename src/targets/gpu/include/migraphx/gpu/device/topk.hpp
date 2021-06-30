#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_TOPK_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_TOPK_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument topk(hipStream_t stream,
              argument val_res,
              argument ind_res,
              argument arg,
              int64_t k,
              int64_t axis,
              bool largest);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
