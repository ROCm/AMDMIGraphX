#include <migraphx/remap.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/squeeze.hpp>
#include <migraphx/op/unsqueeze.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace {
struct find_dot_add
{
    auto matcher() const
    {
        return match::name("add")(match::any_of(
            match::args(match::name("dot")(match::nargs(2)).bind("dot"), match::any().bind("a")),
            match::args(match::used_once().bind("a"),
                        match::name("dot")(match::nargs(2)).bind("dot"))));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins     = r.result;
        auto dot_ins = r.instructions["dot"];
        auto a_ins   = r.instructions["a"];

        auto dot = any_cast<op::dot>(dot_ins->get_operator());

        dot.beta = 1;
        p.replace_instruction(ins, dot, dot_ins->inputs()[0], dot_ins->inputs()[1], a_ins);
    }
};

struct find_1d_conv
{
    auto matcher() const { return match::name("convolution")(match::nargs(2)); }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins = r.result;
        auto op  = any_cast<op::convolution>(ins->get_operator());
        if(op.kdims() != 1)
            return;
        auto x   = ins->inputs()[0];
        auto w   = ins->inputs()[1];
        auto out = ins->outputs();

        auto new_x = p.insert_instruction(std::next(x), op::unsqueeze{{2}}, x);
        auto new_w = p.insert_instruction(std::next(w), op::unsqueeze{{2}}, w);

        op::convolution new_conv = op;
        new_conv.padding.insert(new_conv.padding.begin(), 0);
        new_conv.stride.insert(new_conv.stride.begin(), 1);
        new_conv.dilation.insert(new_conv.dilation.begin(), 1);

        auto new_conv_ins = p.replace_instruction(ins, new_conv, new_x, new_w);
        p.insert_instruction(op::squeeze{{2}}, new_conv_ins);
    }
};

} // namespace

void remap::apply(program& p) const { match::find_matches(p, find_dot_add{}, find_1d_conv{}); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
