#include <migraphx/decompose.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/multibroadcast.hpp>
#include <migraphx/op/mul.hpp>
#include <migraphx/op/add.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace {
struct find_dot_add
{
    auto matcher() const { return match::name("dot")(match::nargs(3)); }

    void apply(program& p, const match::matcher_result& r) const
    {
        auto ins = r.result;
        auto dot = any_cast<op::dot>(ins->get_operator());
        if(not float_equal(dot.beta, 1) and
           not contains({shape::float_type, shape::half_type, shape::double_type},
                        ins->get_shape().type()))
            return;
        auto a_mat = ins->inputs()[0];
        auto b_mat = ins->inputs()[1];
        if(not float_equal(dot.alpha, 1))
        {
            auto alpha = p.add_literal(literal{shape{ins->get_shape().type()}, {dot.alpha}});
            auto alpha_broadcast =
                p.insert_instruction(ins, op::multibroadcast{a_mat->get_shape().lens()}, alpha);
            a_mat = p.insert_instruction(ins, op::mul{}, a_mat, alpha_broadcast);
        }
        auto dot_ins = p.insert_instruction(ins, op::dot{0, 0}, a_mat, b_mat);
        auto c_ins   = ins->inputs()[2];
        if(not float_equal(dot.beta, 1))
        {
            auto beta = p.add_literal(literal{shape{ins->get_shape().type()}, {dot.beta}});
            auto beta_broadcast =
                p.insert_instruction(ins, op::multibroadcast{ins->get_shape().lens()}, beta);
            c_ins = p.insert_instruction(ins, op::mul{}, c_ins, beta_broadcast);
        }
        p.replace_instruction(ins, op::add{}, dot_ins, c_ins);
    }
};

struct find_dot_alpha
{
    auto matcher() const { return match::name("dot")(match::nargs(2)); }

    void apply(program& p, const match::matcher_result& r) const
    {
        auto ins   = r.result;
        auto dot   = any_cast<op::dot>(ins->get_operator());
        auto a_mat = ins->inputs()[0];
        auto b_mat = ins->inputs()[1];
        if(not float_equal(dot.alpha, 1))
        {
            auto alpha = p.add_literal(literal{shape{ins->get_shape().type()}, {dot.alpha}});
            auto alpha_broadcast =
                p.insert_instruction(ins, op::multibroadcast{a_mat->get_shape().lens()}, alpha);
            a_mat = p.insert_instruction(ins, op::mul{}, a_mat, alpha_broadcast);
        }
        p.replace_instruction(ins, op::dot{0, 0}, a_mat, b_mat);
    }
};

} // namespace

void decompose::apply(program& p) const
{
    match::find_matches(p, find_dot_add{}, find_dot_alpha{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
