#ifndef MIGRAPHX_GUARD_RTGLIB_RNN_VARIABLE_SEQ_LENS_HPP
#define MIGRAPHX_GUARD_RTGLIB_RNN_VARIABLE_SEQ_LENS_HPP

#include <migraphx/shape.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/op/rnn_variable_seq_lens.hpp>
#include <migraphx/op/rnn_var_sl_last_output.hpp>
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

struct hip_rnn_var_sl_last_output
{
    op::rnn_var_sl_last_output op;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return migraphx::reflect(self.op, f);
    }

    std::string name() const { return "gpu::" + op.name(); }
    shape compute_shape(std::vector<shape> inputs) const;
    argument compute(context& ctx, const shape&, const std::vector<argument>& args) const;
    std::ptrdiff_t output_alias(const std::vector<shape>& shapes) const
    {
        return shapes.size() - 1;
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
