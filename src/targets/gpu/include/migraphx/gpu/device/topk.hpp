#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_TOPK_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_TOPK_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

argument topk_smallest(hipStream_t stream,
                       const argument& val_res,
                       const argument& ind_res,
                       const argument& arg,
                       int64_t k,
                       int64_t axis);

argument topk_largest(hipStream_t stream,
                      const argument& val_res,
                      const argument& ind_res,
                      const argument& arg,
                      int64_t k,
                      int64_t axis);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
