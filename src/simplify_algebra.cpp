#include <migraphx/simplify_algebra.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/program.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/mul.hpp>
#include <migraphx/op/concat.hpp>
#include <migraphx/op/slice.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/contiguous.hpp>
#include <migraphx/op/as_shape.hpp>
#include <migraphx/op/broadcast.hpp>
#include <migraphx/op/neg.hpp>
#include <migraphx/op/recip.hpp>
#include <migraphx/op/rsqrt.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/algorithm.hpp>

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

MIGRAPHX_PRED_MATCHER(args_has_same_ops, instruction_ref ins)
{
    if(ins->inputs().empty())
        return true;
    return std::all_of(ins->inputs().begin(), ins->inputs().end(), [&](auto j) {
        return j->get_operator() == ins->inputs().front()->get_operator();
    });
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

struct find_concat_unary
{
    auto matcher() const
    {
        return match::name("concat")(args_has_same_ops(),
                                     match::arg(0)(match::nargs(1),
                                                   match::name("relu", "broadcast").bind("x"),
                                                   match::used_once()));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins  = r.result;
        auto x    = r.instructions["x"];
        auto op   = x->get_operator();
        auto axis = any_cast<op::concat>(ins->get_operator()).axis;
        // Adjust broadcast lens
        if(op.name() == "broadcast")
        {
            auto b = any_cast<op::broadcast>(op);
            if(b.axis != axis)
                return;
            b.broadcast_lens = ins->get_shape().lens();
            op               = b;
            axis             = 0;
        }

        auto inputs = ins->inputs();
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto i) {
            return i->inputs().front();
        });
        auto concat = p.insert_instruction(ins, op::concat{axis}, inputs);
        p.replace_instruction(ins, op, concat);
    }
};

struct find_concat_binary
{
    auto matcher() const
    {
        return match::name("concat")(args_has_same_ops(),
                                     match::arg(0)(match::nargs(2),
                                                   match::name("add", "multiply").bind("x"),
                                                   match::used_once()));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins       = r.result;
        auto x         = r.instructions["x"];
        auto op        = x->get_operator();
        auto concat_op = ins->get_operator();

        auto xinputs = ins->inputs();
        std::transform(xinputs.begin(), xinputs.end(), xinputs.begin(), [&](auto i) {
            return i->inputs().front();
        });
        auto yinputs = ins->inputs();
        std::transform(yinputs.begin(), yinputs.end(), yinputs.begin(), [&](auto i) {
            return i->inputs().back();
        });
        auto xconcat = p.insert_instruction(ins, concat_op, xinputs);
        auto yconcat = p.insert_instruction(ins, concat_op, yinputs);
        p.replace_instruction(ins, op, xconcat, yconcat);
    }
};

std::vector<instruction_ref> get_splits(instruction_ref ins)
{
    std::vector<instruction_ref> result;
    std::copy_if(ins->outputs().begin(),
                 ins->outputs().end(),
                 std::back_inserter(result),
                 [&](auto i) { return i->name() == "slice"; });
    if(result.size() < 2)
        return {};
    auto get_slice = [](auto& i) -> auto& { return any_cast<op::slice>(i->get_operator()); };
    auto&& axes    = get_slice(result.front()).axes;
    if(std::any_of(result.begin(), result.end(), [&](auto i) { return get_slice(i).axes != axes; }))
        return {};
    auto get_start = [&](auto& i) -> auto& { return get_slice(i).starts; };
    auto get_end   = [&](auto& i) -> auto& { return get_slice(i).ends; };
    std::sort(
        result.begin(), result.end(), [&](auto x, auto y) { return get_start(x) < get_start(y); });
    if(std::any_of(get_start(result.front()).begin(), get_start(result.front()).end(), [&](auto i) {
           return i != 0;
       }))
        return {};
    auto it = std::adjacent_find(
        result.begin(), result.end(), [&](auto x, auto y) { return get_end(x) != get_start(y); });
    if(it != result.end())
        return {};
    for(std::size_t i = 0; i < axes.size(); i++)
    {
        auto axis = axes[i];
        if(ins->get_shape().lens()[axis] != get_slice(result.back()).ends[i])
            return {};
    }
    return result;
}

struct find_splits
{
    auto matcher() const
    {
        return match::any(match::any_of[match::outputs()](match::name("slice")(
            match::any_of[match::outputs()](match::name("add", "mul", "relu")))));
    }

    static std::vector<std::vector<instruction_ref>>
    get_split_groups(const std::vector<instruction_ref>& splits)
    {
        std::vector<std::vector<instruction_ref>> groups;
        for(auto out : splits.front()->outputs())
        {
            if(out->name() == "slice")
                continue;
            std::vector<instruction_ref> group;
            for(auto split : splits)
            {
                auto it =
                    std::find_if(split->outputs().begin(), split->outputs().end(), [&](auto i) {
                        return i->get_operator() == out->get_operator();
                    });
                if(it == split->outputs().end())
                    break;
                assert((*it)->name() != "slice");
                // If there is a duplicate bail
                if(contains(group, *it))
                    return {};
                group.push_back(*it);
            }
            if(group.size() != splits.size())
                continue;
            groups.push_back(group);
        }
        return groups;
    }

    void apply(program& p, const match::matcher_result& r) const
    {
        auto ins = r.result;

        auto splits = get_splits(ins);
        if(splits.empty())
            return;
        for(const auto& group : get_split_groups(splits))
        {
            auto start = group.front();
            auto op    = start->get_operator();
            if(op.name() == "slice")
                continue;

            // Make sure there is no duplicates
            assert(std::none_of(
                std::next(group.begin()), group.end(), [&](auto i) { return i == start; }));

            auto split_idx    = 0;
            instruction_ref c = p.end();
            if(start->inputs().size() == 1)
            {
                c = p.insert_instruction(std::next(ins), op, ins);
            }
            else if(start->inputs().size() == 2)
            {
                assert(not std::none_of(start->inputs().begin(), start->inputs().end(), [](auto i) {
                    return i->name() == "slice";
                }) && "one argument must be a split");
                auto data_idx = 1;
                if(start->inputs().back()->name() == "slice")
                {
                    split_idx = 1;
                    data_idx  = 0;
                }

                std::vector<instruction_ref> data_args;
                std::transform(group.begin(),
                               group.end(),
                               std::back_inserter(data_args),
                               [&](auto i) { return i->inputs()[data_idx]; });

                // Data arguments must be a constant
                if(std::any_of(data_args.begin(), data_args.end(), [](auto i) {
                       return not i->can_eval();
                   }))
                    return;

                for(auto data : data_args)
                    p.move_instructions(data, ins);

                auto slice_op = any_cast<op::slice>(splits.front()->get_operator());
                assert(not slice_op.axes.empty());
                if(slice_op.axes.size() > 1)
                    return;
                auto concat_axis = slice_op.axes.front();
                // TODO: Check if axises match
                auto concat = p.insert_instruction(ins, op::concat{concat_axis}, data_args);

                std::vector<instruction_ref> args;
                args.resize(2);
                args[split_idx] = ins;
                args[data_idx]  = concat;
                c               = p.insert_instruction(std::next(ins), op, args);
            }
            if(c != p.end())
            {
                for(auto i : group)
                {
                    auto split = i->inputs()[split_idx];
                    assert(split->name() == "slice");
                    // Insert contiguous for reshapes
                    for(auto output : i->outputs())
                    {
                        if(not contains({"reshape", "squeeze", "unsqueeze"}, output->name()))
                            continue;
                        auto x = p.insert_instruction(output, op::contiguous{}, output->inputs());
                        p.replace_instruction(output, output->get_operator(), x);
                    }

                    p.replace_instruction(i, split->get_operator(), c);
                }
            }
        }
    }
};

struct find_split_concat
{
    auto matcher() const
    {
        return match::any(match::any_of[match::outputs()](
            match::name("slice")(match::all_of[match::outputs()](match::name("concat")))));
    }

    void apply(program& p, const match::matcher_result& r) const
    {
        auto ins = r.result;

        auto splits = get_splits(ins);
        if(splits.empty())
            return;
        if(std::any_of(
               splits.begin(), splits.end(), [](auto i) { return i->outputs().size() != 1; }))
            return;
        // Check for concat operator
        auto concat = splits.front()->outputs().front();
        if(std::any_of(splits.begin(), splits.end(), [&](auto i) {
               return i->outputs().front() != concat;
           }))
            return;
        // Check axis match
        auto concat_op = any_cast<op::concat>(concat->get_operator());
        auto split_op  = any_cast<op::slice>(splits.front()->get_operator());
        if(split_op.axes.size() != 1)
            return;
        if(split_op.axes.front() != concat_op.axis)
            return;
        // Replace args
        auto args = concat->inputs();
        auto it =
            std::find_if(args.begin(), args.end(), [&](auto i) { return i == splits.front(); });
        if(std::distance(it, args.end()) < splits.size())
            return;
        // If the slices are not in order then stop
        if(not std::is_sorted(it, it + splits.size(), [](instruction_ref x, instruction_ref y) {
               auto xop = any_cast<op::slice>(x->get_operator());
               auto yop = any_cast<op::slice>(y->get_operator());
               return std::tie(xop.starts, xop.ends) < std::tie(yop.starts, yop.ends);
           }))
            return;
        *it = splits.front()->inputs().front();
        args.erase(std::next(it), it + splits.size());

        if(args.size() == 1)
            p.replace_instruction(concat, args.front());
        else
            p.replace_instruction(concat, concat->get_operator(), args);
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
                {input.lens()[0],
                 input.lens()[1],
                 std::size_t(std::max<std::ptrdiff_t>(1, (input.lens()[2] - 1) / n + 1)),
                 std::size_t(std::max<std::ptrdiff_t>(1, (input.lens()[3] - 1) / n + 1))},
                {input.strides()[0],
                 input.strides()[1],
                 input.strides()[2] * n,
                 input.strides()[3] * n}};
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

MIGRAPHX_PRED_MATCHER(horiz_conv_dot, instruction_ref ins)
{
    auto pred = [&](auto name) {
        return [=](auto i) {
            return i->name() == name and i->inputs().front() == ins and
                   i->inputs().at(1)->can_eval();
        };
    };
    auto dots  = std::count_if(ins->outputs().begin(), ins->outputs().end(), pred("dot"));
    auto convs = std::count_if(ins->outputs().begin(), ins->outputs().end(), pred("convolution"));
    return !(dots < 2 and convs < 2);
}

struct find_conv_dot_horiz_fusion
{
    auto matcher() const { return horiz_conv_dot(); }

    void apply(program& p, const match::matcher_result& r) const
    {
        auto ins = r.result;

        auto pred = [](auto i, auto j) {
            if(i->get_operator() != j->get_operator())
                return false;
            if(not contains({"dot", "convolution"}, i->name()))
                return true;
            auto x = i->inputs()[1]->get_shape().lens();
            auto y = j->inputs()[1]->get_shape().lens();
            if(x.size() != y.size())
                return false;
            // Check that non-axises match
            int axis = 1;
            if(i->name() == "dot")
            {
                axis = x.size() - 1;
            }
            return axis_equal(x, y, axis);
        };

        auto each = [&](auto start, auto last) {
            if(std::distance(start, last) < 2)
                return;
            auto&& name = (*start)->name();
            if(not contains({"dot", "convolution"}, name))
                return;
            auto op   = (*start)->get_operator();
            int group = 1;
            if(name == "convolution")
                group = any_cast<op::convolution>(op).group;
            // Skip group convolution
            if(group != 1)
                return;
            auto input = (*start)->inputs().front();
            std::vector<instruction_ref> args;
            std::transform(
                start, last, std::back_inserter(args), [&](auto x) { return x->inputs().at(1); });
            int axis        = 1;
            int concat_axis = 0;
            if(name == "dot")
            {
                axis        = int(args.front()->get_shape().lens().size() - 1);
                concat_axis = axis;
            }

            for(auto arg : args)
                p.move_instructions(arg, input);
            // TODO: Check if axises match
            auto concat    = p.insert_instruction(input, op::concat{concat_axis}, args);
            auto fused     = p.insert_instruction(std::next(input), op, input, concat);
            int64_t offset = 0;
            for(auto arg : range(start, last))
            {
                int64_t len = arg->get_shape().lens()[axis];
                p.replace_instruction(arg, op::slice{{axis}, {offset}, {offset + len}}, fused);
                offset += len;
            }
        };

        auto outputs = ins->outputs();
        group_by(outputs.begin(), outputs.end(), each, pred);
    }
};

struct find_div_const
{
    auto matcher() const
    {
        return match::name("div")(match::arg(1)(match::is_constant().bind("c")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins   = r.result;
        auto c_ins = r.instructions["c"];

        auto recip = p.insert_instruction(std::next(c_ins), op::recip{}, c_ins);

        auto args = ins->inputs();

        p.replace_instruction(ins, op::mul{}, args.front(), recip);
    }
};

struct find_sub_const
{
    auto matcher() const
    {
        return match::name("sub")(match::arg(1)(match::is_constant().bind("c")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins   = r.result;
        auto c_ins = r.instructions["c"];

        auto neg = p.insert_instruction(std::next(c_ins), op::neg{}, c_ins);

        auto args = ins->inputs();

        p.replace_instruction(ins, op::add{}, args.front(), neg);
    }
};

struct find_rsqrt
{
    auto matcher() const
    {
        return match::name("recip")(match::args(
            match::name("sqrt")(match::used_once(), match::args(match::any().bind("x")))));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins   = r.result;
        auto x_ins = r.instructions["x"];

        p.replace_instruction(ins, op::rsqrt{}, x_ins);
    }
};

void simplify_algebra::apply(program& p) const
{
    // Run simplifications multiple times
    for(int i = 0; i < 8; i++)
    {
        match::find_matches(p,
                            find_inner_broadcast{},
                            find_double_add_lit_broadcast{},
                            find_add_lit_broadcast{},
                            find_add_convs{},
                            find_conv_dot_horiz_fusion{},
                            find_mul_conv{},
                            find_mul_add{},
                            find_div_const{},
                            find_sub_const{},
                            find_rsqrt{},
                            find_concat_unary{},
                            find_concat_binary{},
                            find_split_concat{},
                            find_splits{});
        dead_code_elimination{}.apply(p);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
