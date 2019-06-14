#include <migraphx/eliminate_identity.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void eliminate_identity::apply(program& p) const
{
    auto last = std::prev(p.end());
    for(auto ins : iterator_for(p))
    {
        // Skip the first instruction, since we always process the previous
        // instruction
        if(ins == p.begin())
            continue;
        const auto i = std::prev(ins);

        if(i->name() == "identity")
        {
            p.replace_instruction(i, i->inputs().front());
            p.move_instruction(i, p.end());
        }
        if(ins == last)
        {
            if(ins->name() == "identity")
            {
                const instruction_ref& identity_input = ins->inputs().front();
                if(identity_input->outputs().size() == 1)
                {
                    p.move_instruction(identity_input, i);
                    // since this is the last instruction, removing it only
                    // requires changing "last" and calling remove below
                    last = std::prev(last);
                }
            }
            break;
        }
    }
    p.remove_instructions(std::next(last), p.end());
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
