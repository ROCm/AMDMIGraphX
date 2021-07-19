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
    assert(ins->name() == "quantizelinear");
    auto x       = ins->inputs()[0];
    auto y_scale = ins->inputs()[1];

    if(x->get_shape().type() != y_scale->get_shape().type())
    {
        x = m.insert_instruction(ins, make_op("convert", {{"target_type", shape::float_type}}), x);
    }
    auto div            = m.insert_instruction(ins, make_op("div"), x, y_scale);
    auto add_zero_point = m.insert_instruction(ins, make_op("round"), div);

    if(ins->inputs().size() == 3)
    {
        auto zero_point = m.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::float_type}}), ins->inputs()[2]);
        add_zero_point = m.insert_instruction(ins, make_op("add"), add_zero_point, zero_point);
    }

    int64_t max_quant = 0;
    int64_t min_quant = 0;
    ins->get_shape().visit_type([&](auto qt) {
        max_quant = qt.max();
        min_quant = qt.min();
    });
    auto s = add_zero_point->get_shape();
    std::vector<int> min_data(s.elements(), min_quant);
    std::vector<int> max_data(s.elements(), max_quant);
    auto min_arg = m.add_literal(literal(s, min_data));
    auto max_arg = m.add_literal(literal(s, max_data));

    auto saturate = m.insert_instruction(ins, make_op("clip"), add_zero_point, min_arg, max_arg);
    m.replace_instruction(
        ins, make_op("convert", {{"target_type", ins->get_shape().type()}}), saturate);
}

void apply_dequantizelinear(module& m, instruction_ref ins)
{
    assert(ins->name() == "dequantizelinear");
    auto x = m.insert_instruction(
        ins, make_op("convert", {{"target_type", shape::float_type}}), ins->inputs()[0]);
    auto x_scale = ins->inputs()[1];

    if(ins->inputs().size() == 3)
    {
        auto x_zero_point = m.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::float_type}}), ins->inputs()[2]);
        x = m.insert_instruction(ins, make_op("sub"), x, x_zero_point);
    }

    m.replace_instruction(ins, make_op("mul"), x, x_scale);
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
