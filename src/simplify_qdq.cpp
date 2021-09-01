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

MIGRAPHX_PRED_MATCHER(has_same_value, instruction_ref ins)
{
    if(ins->name() != "@literal")
        return false;
    bool all_same = false;
    ins->get_literal().visit([&](auto s) {
        all_same = std::all_of(s.begin() + 1, s.end(), [&](const auto& scale) {
            return float_equal(scale, s.front());
        });
    });
    return all_same;
}

struct match_find_quantizable_ops
{

    static auto dequantizelinear_op(const std::string& name, const std::string& scale)
    {
        return match::name("dequantizelinear")(
            match::arg(0)(match::skip(match::name("quantizelinear"))(match::any().bind(name))),
            match::arg(1)(match::skip_broadcasts(has_same_value().bind(scale))),
            match::arg(2)(match::skip_broadcasts(match::all_of(match::has_value(0)))));
    }

    auto matcher() const
    {
        return match::name(get_quantizable_op_names())(
            match::arg(0)(dequantizelinear_op("x1", "scale1")),
            match::arg(1)(dequantizelinear_op("x2", "scale2")));
    }

    void apply(module& m, match::matcher_result r) const
    {
        auto qop    = r.result;
        auto q1     = r.instructions["x1"];
        auto q2     = r.instructions["x2"];
        auto scale1 = r.instructions["scale1"];
        auto scale2 = r.instructions["scale2"];

        // Only INT8 type currently supported
        if(q1->get_shape().type() != migraphx::shape::int8_type or
           q2->get_shape().type() != migraphx::shape::int8_type)
            return;

        double scale;
        visit_all(scale1->get_literal(), scale2->get_literal())(
            [&](const auto s1, const auto s2) { scale = s1.front() * s2.front(); });

        auto qop_args  = qop->inputs();
        qop_args.at(0) = q1;
        qop_args.at(1) = q2;
        instruction_ref dq;
        instruction_ref dq_scale;
        instruction_ref zero_point;
        if(qop->name() == "convolution")
        {
            auto conv_val = qop->get_operator().to_value();
            dq            = m.insert_instruction(
                qop, migraphx::make_op("quant_convolution", conv_val), qop_args);
        }
        else if(qop->name() == "dot")
        {
            auto dot_op = any_cast<op::dot>(qop->get_operator());
            if(!(float_equal(dot_op.alpha, 1.0f) and float_equal(dot_op.beta, 0.0f)))
                return;
            if(qop_args.size() == 3)
                qop_args.pop_back();
            dq = m.insert_instruction(
                qop, migraphx::make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), qop_args);
        }
        dq_scale = m.add_literal(static_cast<float>(scale));

        auto lens = dq->get_shape().lens();
        auto scale_mb =
            m.insert_instruction(qop, make_op("multibroadcast", {{"out_lens", lens}}), dq_scale);
        dq = m.insert_instruction(qop, make_op("dequantizelinear"), dq, scale_mb);
        m.replace_instruction(qop, dq);
    }
};

bool compare_literals(instruction_ref ins1, instruction_ref ins2)
{
    if(ins1->name() == "broadcast" or ins1->name() == "multibroadcast")
        ins1 = ins1->inputs().front();
    auto x = ins1->eval();
    if(x.empty())
        return false;
    auto literal1 = ins1->get_literal();
    if(ins2->name() == "broadcast" or ins2->name() == "multibroadcast")
        ins2 = ins2->inputs().front();
    auto y = ins2->eval();
    if(y.empty())
        return false;
    auto literal2 = ins2->get_literal();

    bool diff_shapes_equal_vals = false;
    visit_all(ins1->get_literal(), ins2->get_literal())([&](const auto l1, const auto l2) {
        diff_shapes_equal_vals =
            std::all_of(
                l1.begin() + 1, l1.end(), [&](auto v) { return float_equal(v, l1.front()); }) and
            std::all_of(l2.begin(), l2.end(), [&](auto v) { return float_equal(v, l1.front()); });
    });

    return (x == y) or diff_shapes_equal_vals;
}

void remove_qdq_pairs(module& m)
{
    for(auto ins : iterator_for(m))
    {
        auto args = ins->inputs();
        for(auto&& arg : args)
        {
            if(arg->name() == "dequantizelinear")
            {
                auto q = arg->inputs().front();
                if((q->name() == "quantizelinear") and
                   compare_literals(arg->inputs().at(1), q->inputs().at(1)) and
                   compare_literals(arg->inputs().at(2), q->inputs().at(2)))
                {
                    instruction::replace_argument(ins, arg, q->inputs().front());
                }
            }
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
