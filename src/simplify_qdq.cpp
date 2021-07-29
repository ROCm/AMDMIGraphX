#include <migraphx/simplify_qdq.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/quant_convolution.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/quant_dot.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::unordered_set<std::string> get_quantizable_op_names()
{
    static std::unordered_set<std::string> s = {"convolution", "dot"};
    return s;
}

std::unordered_set<std::string> get_all_op_names()
{
    static auto ops = get_operators();
    static std::unordered_set<std::string> s(ops.begin(), ops.end());
    return s;
}

bool inputs_are_zeros(const instruction_ref& ins)
{
    literal zp_lit;
    if(ins->name() == "multibroadcast" or ins->name() == "broadcast")
        zp_lit = ins->inputs().at(0)->get_literal();
    else
        zp_lit = ins->get_literal();
    std::vector<int64_t> zero_points;
    zp_lit.visit([&](const auto zp) {
        std::transform(
            zp.begin(), zp.end(), std::back_inserter(zero_points), [](auto&& z) { return z; });
    });
    return std::all_of(
        zero_points.begin(), zero_points.end(), [](const auto& z) { return z == 0; });
}

double get_scale(const instruction_ref& ins)
{
    literal scale_lit;
    if(ins->name() == "multibroadcast" or ins->name() == "broadcast")
        scale_lit = ins->inputs().at(0)->get_literal();
    else
        scale_lit = ins->get_literal();
    std::vector<float> scales;
    scale_lit.visit([&](auto sl) {
        std::transform(
            sl.begin(), sl.end(), std::back_inserter(scales), [](auto&& s) { return s; });
    });
    double epsilon = 1e-6;
    if(not std::all_of(scales.begin(), scales.end(), [&](const auto& s) {
           return std::abs(s - scales.front()) < epsilon;
       }))
        MIGRAPHX_THROW("Multiple scales not currently supported");
    return scales.front();
}

instruction_ref insert_quantize_op(module& m,
                                   instruction_ref ins,
                                   const std::string& name,
                                   instruction_ref x,
                                   instruction_ref scale,
                                   instruction_ref shift)
{
    auto lens = x->get_shape().lens();
    auto scale_mb =
        m.insert_instruction(ins, make_op("multibroadcast", {{"output_lens", lens}}), scale);
    auto shift_mb =
        m.insert_instruction(ins, make_op("multibroadcast", {{"output_lens", lens}}), shift);
    return m.insert_instruction(ins, make_op(name), x, scale_mb, shift_mb);
}

struct match_find_quantizable_ops
{
    auto matcher() const
    {
        return match::name(get_all_op_names())(match::arg(0)(
            match::name(get_quantizable_op_names())(
                match::arg(0)(match::name("dequantizelinear")(
                    match::arg(0)(match::name(get_all_op_names()).bind("q0")).bind("dq0"))),
                match::arg(1)(match::name("dequantizelinear")(
                    match::arg(0)(match::name(get_all_op_names()).bind("q1")).bind("dq1"))))
                .bind("qop")));
    }

    void apply(module& m, match::matcher_result r) const
    {
        auto op  = r.result;
        auto qop = r.instructions["qop"];
        auto dq0 = r.instructions["dq0"];
        auto q0  = r.instructions["q0"];
        auto dq1 = r.instructions["dq1"];
        auto q1  = r.instructions["q1"];

        // Only INT8 type currently supported
        if(q0->get_shape().type() != migraphx::shape::int8_type or
           q1->get_shape().type() != migraphx::shape::int8_type)
            return;

        // Only zero_point==0 currently supported
        auto dq0_args  = dq0->inputs();
        auto dq1_args  = dq1->inputs();
        bool all_zeros = true;
        all_zeros &= inputs_are_zeros(dq0_args.at(2));
        all_zeros &= inputs_are_zeros(dq1_args.at(2));
        if(not all_zeros)
            return;

        auto scale     = get_scale(dq0_args.at(1)) * get_scale(dq1_args.at(1));
        auto qop_args  = qop->inputs();
        qop_args.at(0) = q0;
        qop_args.at(1) = q1;
        instruction_ref dq;
        instruction_ref dq_scale;
        instruction_ref zero_point;
        if(qop->name() == "convolution")
        {
            auto conv_op = any_cast<op::convolution>(qop->get_operator());
            qop          = m.insert_instruction(qop,
                                       migraphx::make_op("quant_convolution",
                                                         {{"padding", conv_op.padding},
                                                          {"stride", conv_op.stride},
                                                          {"dilation", conv_op.dilation},
                                                          {"padding_mode", conv_op.padding_mode},
                                                          {"group", conv_op.group}}),
                                       qop_args);
            dq_scale     = m.add_literal(static_cast<float>(scale));
            zero_point   = m.add_literal(0);
        }
        else if(qop->name() == "dot")
        {
            auto dot_op    = any_cast<op::dot>(qop->get_operator());
            auto scale_val = dot_op.alpha / scale;
            auto alpha     = 1;
            auto beta      = (qop->inputs().size() == 3) ? dot_op.beta : 0.0f;
            qop            = m.insert_instruction(
                qop, migraphx::make_op("quant_dot", {{"alpha", alpha}, {"beta", beta}}), qop_args);
            dq_scale   = m.add_literal(static_cast<float>(scale_val));
            zero_point = m.add_literal(static_cast<int>((-1.0f * beta) / scale_val));
        }
        else
        {
            MIGRAPHX_THROW(qop->name() + " does not currently have a quantized implementation.");
        }

        dq              = insert_quantize_op(m, op, "dequantizelinear", qop, dq_scale, zero_point);
        auto op_args    = op->inputs();
        op_args.front() = dq;
        if(op->name() == "@return")
        {
            auto ret = m.add_return({dq});
            m.replace_instruction(op, ret);
        }
        else
        {
            m.replace_instruction(op, op->get_operator(), op_args);
        }
    }
};

void remove_qdq_pairs(module& m)
{
    for(auto ins : iterator_for(m))
    {
        bool replace_op = false;
        auto args       = ins->inputs();
        for(auto&& arg : args)
        {
            if(arg->name() == "dequantizelinear")
            {
                auto q = arg->inputs().front();
                if(q->name() == "quantizelinear")
                {
                    arg        = q->inputs().front();
                    replace_op = true;
                }
            }
        }
        if(replace_op)
        {
            m.replace_instruction(ins, ins->get_operator(), args);
        }
    }
}

void simplify_qdq::apply(module& m) const
{
    match::find_matches(m, match_find_quantizable_ops{});
    migraphx::run_passes(m, {migraphx::dead_code_elimination{}});
    remove_qdq_pairs(m);
    migraphx::run_passes(m, {migraphx::dead_code_elimination{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
