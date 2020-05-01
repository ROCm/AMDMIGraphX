#ifndef MIGRAPHX_GUARD_RTGLIB_RNN_LAST_CELL_OUTPUT_HPP
#define MIGRAPHX_GUARD_RTGLIB_RNN_LAST_CELL_OUTPUT_HPP

#include <migraphx/shape.hpp>
#include <migraphx/op/rnn_last_cell_output.hpp>
#include <migraphx/gpu/rnn_last_output.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_lstm_last_cell_output : dev_rnn_last_output<op::lstm_last_cell_output>
{
    hip_lstm_last_cell_output() {}
    hip_lstm_last_cell_output(op::lstm_last_cell_output o) : dev_rnn_last_output(std::move(o)){};
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
