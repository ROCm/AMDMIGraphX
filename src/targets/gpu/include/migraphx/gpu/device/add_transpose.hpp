#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_ADD_TRANSPOSE_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_ADD_TRANSPOSE_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void add_transpose_arg0(hipStream_t stream,
                        const argument& result,
                        const argument& arg,
                        int slice_start);
void add_transpose_arg1(hipStream_t stream,
                        const argument& result,
                        const argument& arg,
                        int slice_start);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
