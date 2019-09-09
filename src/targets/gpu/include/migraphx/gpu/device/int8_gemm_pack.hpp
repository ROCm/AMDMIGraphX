#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_INT8_GEMM_PACK_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_INT8_GEMM_PACK_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void int8_gemm_pack_a(hipStream_t stream, const argument& result, const argument& arg);

void int8_gemm_pack_b(hipStream_t stream, const argument& result, const argument& arg);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
