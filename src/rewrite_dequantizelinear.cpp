#include <migraphx/rewrite_dequantizelinear.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/dequantizelinear.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>

#include <migraphx/program.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void rewrite_dequantizelinear::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "dequantizelinear")
            continue;

        auto&& op         = any_cast<op::dequantizelinear>(ins->get_operator());
        auto axis         = op.axis;
        auto x            = ins->inputs()[0];
        auto x_scale      = ins->inputs()[1];
        auto x_zero_point = ins->inputs().size() == 3 ? ins->inputs()[2] : m.add_literal(0);

        auto input_lens = x->get_shape().lens();
        int n_dim       = static_cast<int>(input_lens.size());

        instruction_ref zero_point_int32;
        if(x_zero_point->get_shape().elements() != 1)
        {
            auto tuned_axis  = tune_axis(n_dim, axis, ins->name());
            zero_point_int32 = m.insert_instruction(
                ins,
                make_op("broadcast", {{"axis", tuned_axis}, {"dims", input_lens}}),
                x_zero_point);
            zero_point_int32 = m.insert_instruction(
                ins, make_op("convert", {{"target_type", shape::int32_type}}), zero_point_int32);
        }
        else
        {
            zero_point_int32 = m.insert_instruction(
                ins, make_op("convert", {{"target_type", shape::int32_type}}), x_zero_point);
            zero_point_int32 = m.insert_instruction(
                ins, make_op("multibroadcast", {{"output_lens", input_lens}}), zero_point_int32);
        }

        auto sub_zp_int32 =
            m.insert_instruction(ins, make_op("convert", {{"target_type", shape::int32_type}}), x);
        auto sub = m.insert_instruction(ins, make_op("sub"), sub_zp_int32, zero_point_int32);
        auto dequant_input = m.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::float_type}}), sub);

        instruction_ref multiplier;
        if(x_scale->get_shape().elements() != 1)
        {
            auto tuned_axis = tune_axis(n_dim, axis, ins->name());
            multiplier      = m.insert_instruction(
                ins, make_op("broadcast", {{"axis", tuned_axis}, {"dims", input_lens}}), x_scale);
        }
        else
        {
            multiplier = m.insert_instruction(
                ins, make_op("multibroadcast", {{"output_lens", input_lens}}), x_scale);
        }

        m.replace_instruction(ins, make_op("mul"), dequant_input, multiplier);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
