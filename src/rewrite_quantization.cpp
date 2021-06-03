#include <migraphx/rewrite_quantization.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/quantizelinear.hpp>
#include <migraphx/op/dequantizelinear.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>
#include <migraphx/op/broadcast.hpp>
#include <migraphx/op/div.hpp>

#include <migraphx/program.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void rewrite_quantization::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "quantizelinear")
        {
            auto&& op    = any_cast<op::quantizelinear>(ins->get_operator());
            auto axis    = op.axis;
            auto x       = ins->inputs()[0];
            auto y_scale = ins->inputs()[1];

            instruction_ref y_zero_point;
            auto quant_type = shape::int8_type;
            if(ins->inputs().size() == 3)
            {
                y_zero_point = ins->inputs()[2];
                quant_type   = y_zero_point->get_shape().type();
            }
            else
                y_zero_point = m.add_literal(0);

            int max_quant = 255;
            int min_quant = 0;

            if(quant_type == shape::int8_type)
            {
                max_quant = 127;
                min_quant = -128;
            }

            auto max_arg = m.add_literal(max_quant);
            auto min_arg = m.add_literal(min_quant);

            auto input_lens = x->get_shape().lens();
            int n_dim       = static_cast<int>(input_lens.size());

            instruction_ref divisor;
            if(y_scale->get_shape().elements() != 1)
            {
                auto tuned_axis = tune_axis(n_dim, axis, ins->name());
                divisor         = m.insert_instruction(
                    ins, make_op("broadcast", {{"axis", tuned_axis}, {"dims", input_lens}}), y_scale);
            }
            else
            {
                divisor = m.insert_instruction(
                    ins, make_op("multibroadcast", {{"output_lens", input_lens}}), y_scale);
            }

            auto div            = m.insert_instruction(ins, make_op("div"), x, divisor);
            auto add_zero_point = m.insert_instruction(ins, make_op("round"), div);

            instruction_ref zero_point;
            if(y_zero_point->get_shape().elements() != 1)
            {
                auto tuned_axis = tune_axis(n_dim, axis, ins->name());
                zero_point      = m.insert_instruction(
                    ins,
                    make_op("broadcast", {{"axis", tuned_axis}, {"dims", input_lens}}),
                    y_zero_point);
                zero_point = m.insert_instruction(
                    ins, make_op("convert", {{"target_type", shape::int32_type}}), zero_point);
            }
            else
            {
                zero_point = m.insert_instruction(
                    ins, make_op("convert", {{"target_type", shape::int32_type}}), y_zero_point);
                zero_point = m.insert_instruction(
                    ins, make_op("multibroadcast", {{"output_lens", input_lens}}), zero_point);
            }

            add_zero_point = m.insert_instruction(
                ins, make_op("convert", {{"target_type", shape::int32_type}}), add_zero_point);
            auto add = m.insert_instruction(ins, make_op("add"), add_zero_point, zero_point);

            auto s           = add_zero_point->get_shape();
            const auto& lens = s.lens();
            std::vector<int64_t> output_lens(lens.begin(), lens.end());
            if(min_arg->get_shape() != s)
            {
                min_arg = m.insert_instruction(
                    ins, make_op("multibroadcast", {{"output_lens", output_lens}}), min_arg);
            }
            if(max_arg->get_shape() != s)
            {
                max_arg = m.insert_instruction(
                    ins, make_op("multibroadcast", {{"output_lens", output_lens}}), max_arg);
            }

            auto saturate = m.insert_instruction(ins, make_op("clip"), add, min_arg, max_arg);
            m.replace_instruction(ins, make_op("convert", {{"target_type", quant_type}}), saturate);
        }

        else if(ins->name() == "dequantizelinear")
        {
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
                auto tuned_axis = tune_axis(n_dim, axis, ins->name());
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
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
