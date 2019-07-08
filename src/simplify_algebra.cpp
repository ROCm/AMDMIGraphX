#include <migraphx/simplify_algebra.hpp>
#include <migraphx/program.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/mul.hpp>
#include <migraphx/op/broadcast.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/literal.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

auto lit_broadcast()
{
    return match::any_of(match::name("@literal"), match::name("broadcast"));
}
auto not_lit_broadcast()
{
    return match::none_of(match::name("@literal"), match::name("broadcast"));
}
auto op_lit_broadcast(std::string op, std::string x, std::string y)
{
    return match::name(op)(match::either_arg(0, 1)(lit_broadcast().bind(std::move(x)),
                                                      not_lit_broadcast().bind(std::move(y))));
}

struct find_mul_conv
{
    auto matcher() const
    {
        return match::name("mul")(
            match::either_arg(0, 1)(match::name("conv")(match::used_once(), match::args(match::any(), match::is_constant().bind("w"))).bind("conv"), match::name("broadcast").bind("a")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins   = r.result;
        auto conv_ins = r.instructions["conv"];
        auto a_ins = r.instructions["a"];
        auto w_ins = r.instructions["w"];
        
        auto broadcast_op = any_cast<op::broadcast>(a_ins->get_operator());
        if (broadcast_op.axis != 1)
            return;

        auto new_a = p.insert_instruction(ins, op::broadcast{0, w_ins->get_shape().lens()}, a_ins->inputs().front());
        auto new_mul = p.insert_instruction(ins, op::mul{}, new_a, w_ins);
        auto new_conv = p.insert_instruction(ins, conv_ins->get_operator(), conv_ins->inputs().front(), new_mul);
        p.replace_instruction(ins, new_conv);
    }
};

struct find_add_lit_broadcast
{
    auto matcher() const
    {
        return match::name("add")(
            match::args(op_lit_broadcast("add", "a", "x"), op_lit_broadcast("add", "b", "y")));
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

void simplify_algebra::apply(program& p) const { match::find_matches(p, find_add_lit_broadcast{}, find_mul_conv{}); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
