#include <migraphx/constant_propagate.hpp>
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
    if(ins->get_shape().broadcasted() and not ins->get_shape().scalar())
        return true;
    if(ins->get_shape().scalar() and ins->get_shape().elements() != 1)
        return true;
    return false;
}

void constant_propagate::apply(program& p) const
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
