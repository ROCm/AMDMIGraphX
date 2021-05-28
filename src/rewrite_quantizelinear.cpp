#include <migraphx/rewrite_quantizelinear.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/quantizelinear.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>
#include <migraphx/op/broadcast.hpp>
#include <migraphx/op/div.hpp>

#include <migraphx/program.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void rewrite_quantizelinear::apply(module& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->name() != "quantizelinear")
            continue;

        auto&& op = any_cast<op::quantizelinear>(ins->get_operator());
        auto axis = op.axis;

        auto args = ins->inputs();

        int max_quant = 255;
        int min_quant = 0;
        auto quant_type = args[2]->get_shape().type();
        
        if(quant_type == shape::int8_type)
        {
            max_quant = 127;
            min_quant = -128;
        }

        auto max_arg = p.add_literal(max_quant);
        auto min_arg = p.add_literal(min_quant);

        auto input_lens = args[0]->get_shape().lens();
        int n_dim       = static_cast<int>(input_lens.size());

        instruction_ref divisor;
        if (args[1]->get_shape().elements() != 1)
        {
            auto tuned_axis = tune_axis(n_dim, axis, ins->name());
            divisor = p.insert_instruction(ins, make_op("broadcast", {{"axis", tuned_axis}, {"dims", input_lens}}), args[1]);
        }
        else
        {
            divisor = p.insert_instruction(ins, make_op("multibroadcast", {{"output_lens", input_lens}}), args[1]);
        }
        
        auto div = p.insert_instruction(ins, make_op("div"), ins->inputs()[0], divisor);
        auto add_zero_point = p.insert_instruction(ins, make_op("round"), div);

        instruction_ref zero_point;
        if (args[2]->get_shape().elements() != 1)
        {
            auto tuned_axis = tune_axis(n_dim, axis, ins->name());
            zero_point = p.insert_instruction(ins, make_op("broadcast", {{"axis", tuned_axis}, {"dims", input_lens}}), args[2]);
            zero_point = p.insert_instruction(ins, make_op("convert", {{"target_type", shape::int32_type}}), zero_point);
        }
        else
        {
            zero_point = p.insert_instruction(ins, make_op("convert", {{"target_type", shape::int32_type}}), args[2]);
            zero_point = p.insert_instruction(ins, make_op("multibroadcast", {{"output_lens", input_lens}}), zero_point);
        }

        add_zero_point = p.insert_instruction(ins, make_op("convert", {{"target_type", shape::int32_type}}), add_zero_point);
        auto add = p.insert_instruction(ins, make_op("add"), add_zero_point, zero_point);
        
        auto s           = add_zero_point->get_shape();
        const auto& lens = s.lens();
        std::vector<int64_t> output_lens(lens.begin(), lens.end());
        if(min_arg->get_shape() != s)
        {
            min_arg = p.insert_instruction(ins, make_op("multibroadcast", {{"output_lens", output_lens}}), min_arg);
        }
        if(max_arg->get_shape() != s)
        {
            max_arg = p.insert_instruction(ins, make_op("multibroadcast", {{"output_lens", output_lens}}), max_arg);
        }

        auto saturate = p.insert_instruction(ins, make_op("clip"), add, min_arg, max_arg);
        p.replace_instruction(ins, make_op("convert", {{"target_type", quant_type}}), saturate);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
