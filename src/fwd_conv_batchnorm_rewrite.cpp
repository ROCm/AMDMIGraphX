#include <migraph/fwd_conv_batchnorm_rewrite.hpp>
#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/dfor.hpp>

namespace migraph {
void fwd_conv_batchnorm_rewrite::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->op.name() != "batch_norm_inference")
            continue;
        if(not std::all_of(ins->arguments.begin() + 1, ins->arguments.end(), [](auto arg) {
               return arg->op.name() == "@literal";
           }))
            continue;

        auto conv_ins = ins->arguments[0];
        if(conv_ins->op.name() != "convolution")
            continue;
        if(conv_ins->arguments[1]->op.name() != "@literal")
            continue;
        // Get scale, bias, mean, variance from instruction_ref
        const auto& gamma    = ins->arguments[1]->get_literal();
        const auto& bias     = ins->arguments[2]->get_literal();
        const auto& mean     = ins->arguments[3]->get_literal();
        const auto& variance = ins->arguments[4]->get_literal();
        // Get epsilon
        auto bn_op   = any_cast<batch_norm_inference>(ins->op);
        auto epsilon = bn_op.epsilon;
        // Get convolution weights
        const auto& weights = conv_ins->arguments[1]->get_literal();
        // Get convolution op
        auto conv_op      = conv_ins->op;
        auto weights_lens = weights.get_shape().lens();
        auto conv_lens = conv_ins->get_shape().lens();
        argument new_weights{weights.get_shape()};
        argument new_bias{bias.get_shape()};
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
                            gamma2(k) / std::sqrt(variance2(k) + epsilon) * weights2(k, c, h, w);
                    });
                dfor(new_bias.get_shape().elements())(
                    [&](std::size_t c) {
                        new_bias2(c) =
                            bias2(c) - (mean2(c) / std::sqrt(variance2(c) + epsilon));
                    });
            });
        // Replace convolution instruction with updated weights
        auto l_weights = p.add_literal({weights.get_shape(), new_weights.data()});
        auto l_bias    = p.add_literal({new_bias.get_shape(), new_bias.data()});
        auto c = p.replace_instruction(conv_ins, conv_op, {conv_ins->arguments[0], l_weights});
        auto b = p.insert_instruction(ins, broadcast{1}, c, l_bias);
        p.replace_instruction(ins, add{}, {c, b});
    }
}
} // namespace migraph
