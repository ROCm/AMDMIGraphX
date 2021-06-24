#include <migraphx/rewrite_quantization.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>
#include <migraphx/program.hpp>
#include <migraphx/shape.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void apply_quantizelinear(module& m, instruction_ref ins)
{
    auto x       = ins->inputs()[0];
    auto y_scale = ins->inputs()[1];

    auto quant_type   = ins->get_shape().type();
    int64_t max_quant = 0;
    int64_t min_quant = 0;
    ins->get_shape().visit_type([&](auto as) {
        max_quant = as.max();
        min_quant = as.min();
    });
    auto max_arg = m.add_literal(static_cast<int>(max_quant));
    auto min_arg = m.add_literal(static_cast<int>(min_quant));

    auto div            = m.insert_instruction(ins, make_op("div"), x, y_scale);
    auto add_zero_point = m.insert_instruction(ins, make_op("round"), div);
    add_zero_point      = m.insert_instruction(
        ins, make_op("convert", {{"target_type", shape::int32_type}}), add_zero_point);
    if(ins->inputs().size() == 3)
    {
        auto zero_point = ins->inputs()[2];
        zero_point      = m.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::int32_type}}), zero_point);
        add_zero_point = m.insert_instruction(ins, make_op("add"), add_zero_point, zero_point);
    }

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

    auto saturate = m.insert_instruction(ins, make_op("clip"), add_zero_point, min_arg, max_arg);
    m.replace_instruction(ins, make_op("convert", {{"target_type", quant_type}}), saturate);
}

void apply_dequantizelinear(module& m, instruction_ref ins)
{
    auto x       = ins->inputs()[0];
    auto x_scale = ins->inputs()[1];

    instruction_ref dequant_input;
    if(ins->inputs().size() == 3)
    {
        auto x_zero_point     = ins->inputs()[2];
        auto zero_point_int32 = m.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::int32_type}}), x_zero_point);
        auto sub_zp_int32 =
            m.insert_instruction(ins, make_op("convert", {{"target_type", shape::int32_type}}), x);
        x = m.insert_instruction(ins, make_op("sub"), sub_zp_int32, zero_point_int32);
    }

    dequant_input =
        m.insert_instruction(ins, make_op("convert", {{"target_type", shape::float_type}}), x);
    m.replace_instruction(ins, make_op("mul"), dequant_input, x_scale);
}

void rewrite_quantization::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "quantizelinear")
        {
            apply_quantizelinear(m, ins);
        }

        else if(ins->name() == "dequantizelinear")
        {
            apply_dequantizelinear(m, ins);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
