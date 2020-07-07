#include <migraphx/rewrite_batchnorm.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/batch_norm.hpp>
#include <migraphx/op/broadcast.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/mul.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/dfor.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void rewrite_batchnorm::apply(program& p) const
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

        auto s = shape{ins->get_shape().type(), {ins->get_shape().lens()[1]}};
        // Get epsilon
        auto bn_op   = any_cast<op::batch_norm_inference>(ins->get_operator());
        if (bn_op.bn_mode != op::batch_norm_inference::spatial)
            continue;

        auto epsilon = bn_op.epsilon;

        argument a{s};
        argument b{s};
        visit_all(gamma, bias, mean, variance, a, b)(
            [&](auto gamma2, auto bias2, auto mean2, auto variance2, auto a2, auto b2) {
                dfor(a.get_shape().elements())(
                    [&](std::size_t c) { a2[c] = gamma2[c] / std::sqrt(variance2[c] + epsilon); });
                dfor(b.get_shape().elements())([&](std::size_t c) {
                    b2[c] = bias2[c] - (gamma2[c] * mean2[c] / std::sqrt(variance2[c] + epsilon));
                });
            });

        auto broadcast   = op::broadcast{1, ins->get_shape().lens()};
        auto a_ins       = p.add_literal({a.get_shape(), a.data()});
        auto a_broadcast = p.insert_instruction(ins, broadcast, a_ins);
        auto mul         = p.insert_instruction(ins, op::mul{}, ins->inputs().front(), a_broadcast);
        auto b_ins       = p.add_literal({b.get_shape(), b.data()});
        auto b_broadcast = p.insert_instruction(ins, broadcast, b_ins);
        auto add         = p.insert_instruction(ins, op::add{}, mul, b_broadcast);
        p.replace_instruction(ins, add);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
