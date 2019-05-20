#include <migraphx/fwd_conv_batchnorm_rewrite.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/batch_norm.hpp>
#include <migraphx/op/broadcast.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/dfor.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void fwd_conv_batchnorm_rewrite::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->name() != "batch_norm_inference")
            continue;
        // Get scale, bias, mean, variance from inputs
        auto gamma    = ins->inputs()[1]->eval();
        auto bias     = ins->inputs()[2]->eval();
        auto mean     = ins->inputs()[3]->eval();
        auto variance = ins->inputs()[4]->eval();
        if(any_of({gamma, bias, mean, variance}, [](auto arg) { return arg.empty(); }))
            continue;

        auto conv_ins = ins->inputs()[0];
        if(conv_ins->name() != "convolution")
            continue;
        // Get convolution weights
        auto weights = conv_ins->inputs()[1]->eval();
        if(weights.empty())
            continue;
        // Get epsilon
        auto bn_op   = any_cast<op::batch_norm_inference>(ins->get_operator());
        auto epsilon = bn_op.epsilon;
        // Get convolution op
        auto conv_op      = conv_ins->get_operator();
        auto weights_lens = weights.get_shape().lens();
        auto conv_lens    = conv_ins->get_shape().lens();
        argument new_weights{weights.get_shape()};
        argument new_bias{{bias.get_shape().type(), {bias.get_shape().elements()}}};
        visit_all(weights, gamma, bias, mean, variance, new_weights, new_bias)(
            [&](auto weights2,
                auto gamma2,
                auto bias2,
                auto mean2,
                auto variance2,
                auto new_weights2,
                auto new_bias2) {
                dfor(weights_lens[0], weights_lens[1], weights_lens[2], weights_lens[3])(
                    [&](std::size_t k, std::size_t c, std::size_t h, std::size_t w) {
                        new_weights2(k, c, h, w) =
                            gamma2[k] / std::sqrt(variance2[k] + epsilon) * weights2(k, c, h, w);
                    });
                dfor(new_bias.get_shape().elements())([&](std::size_t c) {
                    new_bias2[c] =
                        bias2[c] - (gamma2[c] * mean2[c] / std::sqrt(variance2[c] + epsilon));
                });
            });
        // Replace convolution instruction with updated weights
        auto l_weights = p.add_literal({weights.get_shape(), new_weights.data()});
        auto l_bias    = p.add_literal({new_bias.get_shape(), new_bias.data()});
        auto c = p.replace_instruction(conv_ins, conv_op, {conv_ins->inputs()[0], l_weights});
        auto b = p.insert_instruction(ins, op::broadcast{1, c->get_shape().lens()}, l_bias);
        p.replace_instruction(ins, op::add{}, {c, b});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
