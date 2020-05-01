#ifndef MIGRAPHX_GUARD_RTGLIB_RNN_LAST_HS_OUTPUT_HPP
#define MIGRAPHX_GUARD_RTGLIB_RNN_LAST_HS_OUTPUT_HPP

#include <migraphx/shape.hpp>
#include <migraphx/op/rnn_last_hs_output.hpp>
#include <migraphx/gpu/rnn_last_output.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_rnn_last_hs_output : dev_rnn_last_output<op::rnn_last_hs_output>
{
    hip_rnn_last_hs_output() {}
    hip_rnn_last_hs_output(op::rnn_last_hs_output o) : dev_rnn_last_output(std::move(o)) {};
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
