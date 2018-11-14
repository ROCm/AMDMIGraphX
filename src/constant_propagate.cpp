#include <migraphx/constant_propagate.hpp>
#include <migraphx/program.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/literal.hpp>

namespace migraphx {
inline namespace MIGRAPH_INLINE_NS {

struct match_const_add
{
    auto matcher() const
    {
        return match::name("add")(match::args(match::name("@literal"), match::name("@literal")));
    }

    void apply(program& p, match::matcher_result r) const
    {
        auto ins  = r.result;
        auto arg1 = ins->inputs().at(0)->get_literal();
        auto arg2 = ins->inputs().at(1)->get_literal();

        auto sum = p.add_literal(transform(arg1, arg2, [](auto x, auto y) { return x + y; }));
        p.replace_instruction(ins, sum);
    }
};

void constant_propagate::apply(program& p) const { match::find_matches(p, match_const_add{}); }

} // namespace MIGRAPH_INLINE_NS
} // namespace migraphx
