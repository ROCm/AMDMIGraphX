#ifndef MIGRAPHX_GUARD_RTGLIB_DEVICE_RNN_VARIABLE_SEQUENCES_HPP
#define MIGRAPHX_GUARD_RTGLIB_DEVICE_RNN_VARIABLE_SEQUENCES_HPP

#include <migraphx/argument.hpp>
#include <migraphx/config.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
namespace device {

void rnn_shift_hidden_states(hipStream_t stream,
                             const argument& result,
                             const argument& arg_hs,
                             const argument& arg_sl,
                             bool is_reverse);

void rnn_shift_sequences(hipStream_t stream,
                         const argument& result,
                         const argument& arg_hs,
                         const argument& arg_sl);

} // namespace device
} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
