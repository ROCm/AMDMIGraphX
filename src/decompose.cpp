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
        auto dot_ins =
            p.insert_instruction(ins, op::dot{dot.alpha, 0}, ins->inputs()[0], ins->inputs()[1]);
        auto c_ins = ins->inputs()[2];
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

} // namespace

void decompose::apply(program& p) const { match::find_matches(p, find_dot_add{}); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
