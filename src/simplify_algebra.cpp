#include <migraphx/simplify_algebra.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/program.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/mul.hpp>
#include <migraphx/op/concat.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/as_shape.hpp>
#include <migraphx/op/broadcast.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/literal.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

auto lit_broadcast() { return match::any_of(match::is_constant(), match::name("broadcast")); }
auto not_lit_broadcast() { return match::none_of(match::is_constant(), match::name("broadcast")); }
auto op_lit_broadcast(std::string op, std::string x, std::string y)
{
    return match::name(std::move(op))(match::either_arg(0, 1)(
        lit_broadcast().bind(std::move(x)), not_lit_broadcast().bind(std::move(y))));
}

auto conv_const_weights()
{
    return match::name("convolution")(match::used_once(),
                                      match::args(match::any(), match::is_constant().bind("w")));
}

struct find_mul_conv
{
    auto matcher() const
    {
        return match::name("mul")(match::either_arg(0, 1)(conv_const_weights().bind("conv"),
                                                          match::name("broadcast").bind("a")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins      = r.result;
        auto conv_ins = r.instructions["conv"];
        auto a_ins    = r.instructions["a"];
        auto w_ins    = r.instructions["w"];

        auto broadcast_op = any_cast<op::broadcast>(a_ins->get_operator());
        if(broadcast_op.axis != 1)
            return;

        auto new_a = p.insert_instruction(
            ins, op::broadcast{0, w_ins->get_shape().lens()}, a_ins->inputs().front());
        auto new_mul  = p.insert_instruction(ins, op::mul{}, new_a, w_ins);
        auto new_conv = p.insert_instruction(
            ins, conv_ins->get_operator(), conv_ins->inputs().front(), new_mul);
        p.replace_instruction(ins, new_conv);
    }
};

// a * (x + b) => a * x + a * b
struct find_mul_add
{
    auto matcher() const
    {
        return match::name("mul")(match::either_arg(0, 1)(
            match::name("add")(
                match::either_arg(0, 1)(
                    match::any().bind("x"),
                    match::any_of(conv_const_weights(), match::is_constant()).bind("b")),
                match::none_of(match::args(match::is_constant(), match::is_constant())),
                match::used_once()),
            match::is_constant().bind("a")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins   = r.result;
        auto a_ins = r.instructions["a"];
        auto b_ins = r.instructions["b"];
        auto x_ins = r.instructions["x"];
        assert(x_ins != b_ins);

        auto ax_ins = p.insert_instruction(ins, op::mul{}, a_ins, x_ins);
        auto ab_ins = p.insert_instruction(ins, op::mul{}, a_ins, b_ins);
        p.replace_instruction(ins, op::add{}, ax_ins, ab_ins);
    }
};

struct find_add_lit_broadcast
{
    auto matcher() const
    {
        return match::name("add")(
            match::either_arg(0, 1)(op_lit_broadcast("add", "a", "x"), lit_broadcast().bind("b")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["x"];
        auto a_ins = r.instructions["a"];
        auto b_ins = r.instructions["b"];

        auto sumab = p.insert_instruction(ins, op::add{}, a_ins, b_ins);
        p.replace_instruction(ins, op::add{}, x_ins, sumab);
    }
};

struct find_double_add_lit_broadcast
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

        instruction_ref sumab;

        if(a_ins->name() == "broadcast" and b_ins->name() == "broadcast")
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

struct find_inner_broadcast
{
    auto matcher() const
    {
        return match::name("mul", "add")(
            match::args(match::name("broadcast").bind("x"), match::name("broadcast").bind("y")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["x"];
        auto y_ins = r.instructions["y"];

        auto xbroadcast = any_cast<op::broadcast>(x_ins->get_operator());
        auto ybroadcast = any_cast<op::broadcast>(y_ins->get_operator());

        if(xbroadcast.axis != ybroadcast.axis)
            return;

        auto op = p.insert_instruction(
            ins, ins->get_operator(), x_ins->inputs().front(), y_ins->inputs().front());
        p.replace_instruction(ins, xbroadcast, op);
    }
};

bool axis_equal(const std::vector<std::size_t>& x,
                const std::vector<std::size_t>& y,
                std::size_t axis)
{
    return x.size() == y.size() and x.size() > axis and
           std::equal(x.begin(), x.begin() + axis, y.begin()) and
           std::equal(x.begin() + axis + 1, x.end(), y.begin() + axis + 1);
}

bool axis_shape_equal(const shape& x, const shape& y, std::size_t axis)
{
    // TODO: Check strides
    return axis_equal(x.lens(), y.lens(), axis);
}

struct find_add_convs
{
    auto matcher() const
    {
        return match::name("add")(
            match::args(conv_const_weights().bind("a"), conv_const_weights().bind("b")));
    }

    static bool symmetrical_strides(const op::convolution& op)
    {
        return op.stride[0] == op.stride[1];
    }

    static std::size_t compute_stride_factor(const op::convolution& x, const op::convolution& y)
    {
        if(not symmetrical_strides(x))
            return 0;
        if(not symmetrical_strides(y))
            return 0;
        if((x.stride[0] % y.stride[0]) != 0)
            return 0;
        return x.stride[0] / y.stride[0];
    }

    static shape compute_stride_shape(const shape& input, std::size_t n)
    {
        return {input.type(),
                {input.lens()[0], input.lens()[1], input.lens()[2] / n, input.lens()[3] / n}, 
                {input.strides()[0], input.strides()[1], input.strides()[2] * n, input.strides()[3] * n}};
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins       = r.result;
        auto a_conv    = r.instructions["a"];
        auto a_input   = a_conv->inputs().at(0);
        auto a_weights = a_conv->inputs().at(1);
        auto b_conv    = r.instructions["b"];
        auto b_input   = b_conv->inputs().at(0);
        auto b_weights = b_conv->inputs().at(1);

        if(not axis_shape_equal(a_weights->get_shape(), b_weights->get_shape(), 1))
            return;

        auto a_op   = any_cast<op::convolution>(a_conv->get_operator());
        auto b_op   = any_cast<op::convolution>(b_conv->get_operator());
        auto new_op = a_op;

        if(a_op != b_op)
        {
            if(std::tie(a_op.padding, a_op.dilation, a_op.group) ==
                   std::tie(b_op.padding, b_op.dilation, b_op.group) and
               a_weights->get_shape().lens()[2] == 1 and a_weights->get_shape().lens()[3] == 1)
            {
                if(a_op.stride < b_op.stride)
                {
                    auto n = compute_stride_factor(b_op, a_op);
                    if(n == 0)
                        return;
                    new_op  = a_op;
                    b_input = p.insert_instruction(
                        ins, op::as_shape{compute_stride_shape(b_input->get_shape(), n)}, b_input);
                }
                else if(b_op.stride < a_op.stride)
                {
                    auto n = compute_stride_factor(a_op, b_op);
                    if(n == 0)
                        return;
                    new_op  = b_op;
                    a_input = p.insert_instruction(
                        ins, op::as_shape{compute_stride_shape(a_input->get_shape(), n)}, a_input);
                }
                else
                    return;
            }
            else
                return;
        }

        auto concat_input   = p.insert_instruction(ins, op::concat{1}, a_input, b_input);
        auto concat_weights = p.insert_instruction(ins, op::concat{1}, a_weights, b_weights);
        p.replace_instruction(ins, new_op, concat_input, concat_weights);
    }
};

void simplify_algebra::apply(program& p) const
{
    // Run simplifications multiple times
    for(int i = 0; i < 4; i++)
    {
        match::find_matches(p,
                            find_inner_broadcast{},
                            find_double_add_lit_broadcast{},
                            find_add_lit_broadcast{},
                            find_add_convs{},
                            find_mul_conv{},
                            find_mul_add{});
        dead_code_elimination{}.apply(p);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
