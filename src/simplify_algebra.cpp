#include <migraphx/simplify_algebra.hpp>
#include <migraphx/program.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/literal.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct find_add_lit_broadcast
{
    auto lit_broadcast() const
    {
        return match::any_of(match::name("@literal"), match::name("broadcast"));
    }
    auto not_lit_broadcast() const
    {
        return match::none_of(match::name("@literal"), match::name("broadcast"));
    }
    auto add_lit_broadcast(std::string x, std::string y) const
    {
        return match::name("add")(match::either_arg(0, 1)(lit_broadcast().bind(std::move(x)),
                                                          not_lit_broadcast().bind(std::move(y))));
    }
    auto matcher() const
    {
        return match::name("add")(
            match::args(add_lit_broadcast("a", "x"), add_lit_broadcast("b", "y")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["x"];
        auto y_ins = r.instructions["y"];
        auto a_ins = r.instructions["a"];
        auto b_ins = r.instructions["b"];

        if(a_ins->name() != b_ins->name())
            return;
        instruction_ref sumab;

        if(a_ins->name() == "broadcast")
        {
            if(a_ins->inputs().at(0)->get_shape() != b_ins->inputs().at(0)->get_shape())
                return;
            auto op = a_ins->get_operator();
            auto presum =
                p.insert_instruction(ins, op::add{}, a_ins->inputs().at(0), b_ins->inputs().at(0));
            sumab = p.insert_instruction(ins, op, presum);
        }
        else
        {
            sumab = p.insert_instruction(ins, op::add{}, a_ins, b_ins);
        }

        auto sumxy = p.insert_instruction(ins, op::add{}, x_ins, y_ins);
        p.replace_instruction(ins, op::add{}, sumxy, sumab);
    }
};

void simplify_algebra::apply(program& p) const { match::find_matches(p, find_add_lit_broadcast{}); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
