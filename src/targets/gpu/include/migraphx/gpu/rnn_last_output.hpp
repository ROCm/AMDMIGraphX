#ifndef MIGRAPHX_GUARD_RTGLIB_RNN_LAST_OUTPUT_HPP
#define MIGRAPHX_GUARD_RTGLIB_RNN_LAST_OUTPUT_HPP

#include <migraphx/shape.hpp>
#include <migraphx/op/rnn_last_output.hpp>
#include <migraphx/gpu/device/rnn_last_output.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

template<class Op>
struct dev_rnn_last_output
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

    argument
    compute(context& ctx, const shape&, const std::vector<argument>& args) const
    {
        device::rnn_last_output(ctx.get_stream().get(),
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

    dev_rnn_last_output() {}
    dev_rnn_last_output(Op o) : op(o) {}
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
