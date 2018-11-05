#include <migraph/constant_propagate.hpp>
#include <migraph/program.hpp>
#include <migraph/matcher.hpp>
#include <migraph/literal.hpp>

namespace migraph {
inline namespace version_1 {

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

} // namespace version_1
} // namespace migraph
