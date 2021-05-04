#include <migraphx/decompose.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace {

struct alpha_beta
{
    float alpha = 0.0;
    float beta  = 0.0;
};

alpha_beta get_alpha_beta(const operation& op)
{
    auto v = op.to_value();
    return {v.at("alpha").to<float>(), v.at("beta").to<float>()};
}

struct find_dot_add
{
    auto matcher() const { return match::name("dot", "quant_dot")(match::nargs(3)); }

    void apply(module& p, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto dot   = get_alpha_beta(ins->get_operator());
        auto a_ins = ins->inputs()[0];
        auto b_ins = ins->inputs()[1];
        if(not float_equal(dot.alpha, 1))
        {
            auto alpha = p.add_literal(literal{shape{a_ins->get_shape().type()}, {dot.alpha}});
            auto alpha_broadcast = p.insert_instruction(
                ins,
                make_op("multibroadcast", {{"output_lens", a_ins->get_shape().lens()}}),
                alpha);
            a_ins = p.insert_instruction(ins, make_op("mul"), a_ins, alpha_broadcast);
        }
        auto dot_ins = p.insert_instruction(ins, make_op(ins->name(), {{"beta", 0}}), a_ins, b_ins);

        auto c_ins = ins->inputs()[2];
        if(not float_equal(dot.beta, 1))
        {
            auto beta = p.add_literal(literal{shape{c_ins->get_shape().type()}, {dot.beta}});
            auto beta_broadcast = p.insert_instruction(
                ins, make_op("multibroadcast", {{"output_lens", ins->get_shape().lens()}}), beta);
            c_ins = p.insert_instruction(ins, make_op("mul"), c_ins, beta_broadcast);
        }
        p.replace_instruction(ins, make_op("add"), dot_ins, c_ins);
    }
};

struct find_dot_alpha
{
    auto matcher() const { return match::name("dot", "quant_dot")(match::nargs(2)); }

    void apply(module& p, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto dot   = get_alpha_beta(ins->get_operator());
        auto a_ins = ins->inputs()[0];
        auto b_ins = ins->inputs()[1];
        if(not float_equal(dot.alpha, 1))
        {
            auto alpha = p.add_literal(literal{shape{a_ins->get_shape().type()}, {dot.alpha}});
            auto alpha_broadcast = p.insert_instruction(
                ins,
                make_op("multibroadcast", {{"output_lens", a_ins->get_shape().lens()}}),
                alpha);
            a_ins = p.insert_instruction(ins, make_op("mul"), a_ins, alpha_broadcast);
        }
        p.replace_instruction(ins, make_op(ins->name(), {{"beta", 0}}), a_ins, b_ins);
    }
};

} // namespace

void decompose::apply(module& p) const { match::find_matches(p, find_dot_add{}, find_dot_alpha{}); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
