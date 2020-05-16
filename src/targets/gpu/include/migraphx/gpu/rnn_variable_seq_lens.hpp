#ifndef MIGRAPHX_GUARD_RTGLIB_RNN_VARIABLE_SEQ_LENS_HPP
#define MIGRAPHX_GUARD_RTGLIB_RNN_VARIABLE_SEQ_LENS_HPP

#include <migraphx/shape.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/op/rnn_variable_seq_lens.hpp>
#include <migraphx/gpu/device/rnn_variable_seq_lens.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct hip_rnn_var_sl_shift_sequence
{
    op::rnn_var_sl_shift_sequence op;

    std::string name() const { return "gpu::rnn_var_sl_shift_sequence"; }
    shape compute_shape(std::vector<shape> inputs) const;
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

struct hip_rnn_var_sl_shift_output
{
    op::rnn_var_sl_shift_output op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "gpu::rnn_var_sl_shift_output"; }
    shape compute_shape(std::vector<shape> inputs) const;
    argument
    compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

template <class Op>
struct hip_rnn_var_sl_last_output
{
    Op op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "gpu::" + op.name(); }
    shape compute_shape(std::vector<shape> inputs) const
    {
        inputs.pop_back();
        return op.compute_shape(inputs);
    }

    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        device::rnn_var_sl_last_output(ctx.get_stream().get(),
                                       args.back(),
                                       args.at(0),
                                       args.at(1),
                                       (op.direction == op::rnn_direction::reverse));
        return args.back();
    }

    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }

    // dev_rnn_last_output() {}
    // dev_rnn_last_output(Op o) : op(o) {}
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
