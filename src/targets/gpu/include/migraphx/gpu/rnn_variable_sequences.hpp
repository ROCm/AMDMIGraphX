#ifndef MIGRAPHX_GUARD_RTGLIB_RNN_VARIABLE_SEQUENCES_HPP
#define MIGRAPHX_GUARD_RTGLIB_RNN_VARIABLE_SEQUENCES_HPP

#include <migraphx/shape.hpp>
#include <migraphx/op/rnn_variable_sequences.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct hip_rnn_clear_missing_frames
{
    op::rnn_clear_missing_frames op;

    std::string name() const { return "gpu::rnn_clear_missing_frames"; }
    shape compute_shape(std::vector<shape> inputs) const;
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
