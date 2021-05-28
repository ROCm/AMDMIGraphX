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

void rewrite_dequantizelinear::apply(module& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->name() != "dequantizelinear")
            continue;

        auto&& op = any_cast<op::dequantizelinear>(ins->get_operator());
        auto axis = op.axis;

        auto args = ins->inputs();

        auto input_lens = args[0]->get_shape().lens();
        int n_dim       = static_cast<int>(input_lens.size());

        instruction_ref zero_point_int32;
        if (args[2]->get_shape().elements() != 1)
        {
            auto tuned_axis = tune_axis(n_dim, axis, ins->name());
            zero_point_int32 = p.insert_instruction(ins, make_op("broadcast", {{"axis", tuned_axis}, {"dims", input_lens}}), args[2]);
            zero_point_int32 = p.insert_instruction(ins, make_op("convert", {{"target_type", shape::int32_type}}), zero_point_int32);
        }
        else
        {
            zero_point_int32 = p.insert_instruction(ins, make_op("convert", {{"target_type", shape::int32_type}}), args[2]);
            zero_point_int32 = p.insert_instruction(ins, make_op("multibroadcast", {{"output_lens", input_lens}}), zero_point_int32);
        }

        auto sub_zp_int32 = p.insert_instruction(ins, make_op("convert", {{"target_type", shape::int32_type}}), args[0]);
        auto sub = p.insert_instruction(ins, make_op("sub"), sub_zp_int32, zero_point_int32);
        auto dequant_input = p.insert_instruction(ins, make_op("convert", {{"target_type", shape::float_type}}), sub);

        instruction_ref multiplier;
        if (args[1]->get_shape().elements() != 1)
        {
            auto tuned_axis = tune_axis(n_dim, axis, ins->name());
            multiplier = p.insert_instruction(ins, make_op("broadcast", {{"axis", tuned_axis}, {"dims", input_lens}}), args[1]);
        }
        else
        {
            multiplier = p.insert_instruction(ins, make_op("multibroadcast", {{"output_lens", input_lens}}), args[1]);
        }

        p.replace_instruction(ins, make_op("mul"), dequant_input, multiplier);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
