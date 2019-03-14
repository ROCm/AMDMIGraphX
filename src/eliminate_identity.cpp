#include <migraphx/eliminate_identity.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void eliminate_identity::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->get_operator().name() == "identity")
        {
            if(ins != p.end())
            {
                instruction_ref identity_input{ins->inputs().at(0)};
                auto next_ins = std::next(ins);
                std::vector<instruction_ref> next_ins_inputs{next_ins->inputs()};
                std::replace_if(next_ins_inputs.begin(),
                                next_ins_inputs.end(),
                                [&](instruction_ref& input) { return input == ins; },
                                identity_input);
                p.replace_instruction(next_ins, next_ins->get_operator(), next_ins_inputs);
            }
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
