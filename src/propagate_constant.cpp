#include <migraphx/propagate_constant.hpp>
#include <migraphx/program.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/functional.hpp>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool skip_propogate(instruction_ref ins)
{
    if (ins->name() == "contiguous")
        return skip_propogate(ins->inputs().front());
    auto&& s = ins->get_shape();
    if(s.broadcasted() and not s.scalar())
        return true;
    if(s.scalar() and s.elements() != 1)
        return true;
    return false;
}

void propagate_constant::apply(program& p) const
{
    for(auto i : iterator_for(p))
    {
        if(i->name() != "@literal")
            continue;
        if(i->outputs().empty())
            continue;
        fix([&](auto self, auto ins) {
            std::unordered_set<instruction_ref> children(ins->outputs().begin(),
                                                         ins->outputs().end());
            for(auto child : children)
            {
                if(child->name() == "@literal" or skip_propogate(child))
                {
                    self(child);
                    continue;
                }
                auto r = child->eval();
                if(not r.empty())
                {
                    assert(r.get_shape() == child->get_shape());
                    auto l = p.add_literal(r.get_shape(), r.data());
                    self(p.replace_instruction(child, l));
                }
            }
        })(i);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
