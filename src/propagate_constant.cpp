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
    if(ins->name() == "@literal")
        return true;
    auto&& s = ins->get_shape();
    if(s.broadcasted() and not s.scalar())
        return true;
    if(s.scalar() and s.elements() != 1)
        return true;
    return false;
}

void propagate_constant::apply(program& p) const
{
    fix([&](auto self, auto ins) {
        if(not skip_propogate(ins))
        {
            auto r = ins->eval();
            if(not r.empty())
            {
                assert(r.get_shape() == ins->get_shape());
                auto l = p.add_literal(r.get_shape(), r.data());
                p.replace_instruction(ins, l);
                return;
            }
        }
        std::unordered_set<instruction_ref> children(ins->inputs().begin(), ins->inputs().end());
        for(auto child : children)
            self(child);
    })(std::prev(p.end()));
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
