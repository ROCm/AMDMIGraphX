#include <migraphx/gpu/rnn_variable_sequences.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/rnn_variable_sequences.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_rnn_clear_missing_frames::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_rnn_clear_missing_frames::compute(context& ctx,
                                               const shape&,
                                               const std::vector<argument>& args) const
{
    device::rnn_clear_missing_frames(ctx.get_stream().get(), args.back(), args.at(0), args.at(1));
    return args.back();
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
