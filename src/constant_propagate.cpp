#include <migraphx/constant_propagate.hpp>
#include <migraphx/program.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/functional.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct match_const_add
{
    auto matcher() const
    {
        return match::name("add")(match::args(match::name("@literal"), match::name("@literal")));
    }

    void apply(program& p, const match::matcher_result& r) const
    {
        auto ins  = r.result;
        auto arg1 = ins->inputs().at(0)->get_literal();
        auto arg2 = ins->inputs().at(1)->get_literal();

        auto sum = p.add_literal(transform(arg1, arg2, [](auto x, auto y) { return x + y; }));
        p.replace_instruction(ins, sum);
    }
};

void constant_propagate::apply(program& p) const 
{
    fix([&](auto self, auto ins) {
        if (not ins->get_shape().broadcasted())
        {
            auto r = ins->eval();
            if (not r.empty())
            {
                auto l = p.add_literal(r.get_shape(), r.data());
                p.replace_instruction(ins, l);
                return;
            }
        }
        auto children = ins->inputs();
        for(auto child:children)
            self(child);
    })(std::prev(p.end()));
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
