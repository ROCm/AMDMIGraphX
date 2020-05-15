#include <migraphx/gpu/rnn_variable_seq_lens.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device/rnn_variable_seq_lens.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

shape hip_rnn_var_sl_shift_output::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_rnn_var_sl_shift_output::compute(context& ctx,
                                              const shape&,
                                              const std::vector<argument>& args) const
{
    device::rnn_var_sl_shift_output(ctx.get_stream().get(),
                                    args.back(),
                                    args.at(0),
                                    args.at(1),
                                    (op.direction == op::rnn_direction::reverse));
    return args.back();
}

shape hip_rnn_var_sl_shift_sequence::compute_shape(std::vector<shape> inputs) const
{
    inputs.pop_back();
    return op.compute_shape(inputs);
}

argument hip_rnn_var_sl_shift_sequence::compute(context& ctx,
                                                const shape&,
                                                const std::vector<argument>& args) const
{
    device::rnn_var_sl_shift_sequence(ctx.get_stream().get(), args.back(), args.at(0), args.at(1));
    return args.back();
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
