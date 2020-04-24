#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_RNN_CLEAR_MISSING_FRAMES_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_RNN_CLEAR_MISSING_FRAMES_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void rnn_clear_missing_frames(hipStream_t stream,
                              const argument& result,
                              const argument& arg_hs,
                              const argument& arg_sl)

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
