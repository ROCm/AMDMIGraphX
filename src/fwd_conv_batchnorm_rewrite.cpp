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
        if(ins->op.name() == "batch_norm_inference")
        {
            auto ins_prev = ins->arguments[0];
            if(ins_prev->op.name() == "convolution")
            {
                // Get scale, bias, mean, variance from instruction_ref
                auto gamma    = ins->arguments[1]->lit;
                auto bias     = ins->arguments[2]->lit;
                auto mean     = ins->arguments[3]->lit;
                auto variance = ins->arguments[4]->lit;
                // Get epsilon
                auto bn_op   = any_cast<batch_norm_inference>(ins->op);
                auto epsilon = bn_op.epsilon;
                // Get convolution weights
                auto weights = ins_prev->arguments[1]->lit;
                // Get convolution op
                auto conv_op      = ins_prev->op;
                auto out_channels = weights.get_shape().lens()[0];
                auto in_channels  = weights.get_shape().lens()[1];
                auto height       = weights.get_shape().lens()[2];
                auto width        = weights.get_shape().lens()[3];
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
                        dfor(out_channels, in_channels, height, width)(
                            [&](std::size_t k, std::size_t c, std::size_t h, std::size_t w) {
                                new_weights2(k, c, h, w) = gamma2(k) /
                                                           std::sqrt(variance2(k) + epsilon) *
                                                           weights2(k, c, h, w);
                                new_bias2(k, c, h, w) =
                                    bias2(k) - (mean2(k) / std::sqrt(variance2(k) + epsilon));
                            });
                    });
                // Replace convolution instruction with updated weights
                auto l_weights = p.add_literal({weights.get_shape(), new_weights.data()});
                auto l_bias    = p.add_literal({bias.get_shape(), new_bias.data()});
                auto c =
                    p.replace_instruction(ins_prev, conv_op, {ins_prev->arguments[0], l_weights});
                p.replace_instruction(ins, add{}, {c, l_bias});
            }
        }
    }
}
} // namespace migraph
