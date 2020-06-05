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
} // namespace

void remap::apply(program& p) const { match::find_matches(p, find_dot_add{}); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
