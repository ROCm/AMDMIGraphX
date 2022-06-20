#include <migraphx/eliminate_identity.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void eliminate_identity::apply(module& m) const
{
    auto last = std::prev(m.end());
    for(auto ins : iterator_for(m))
    {
        // Skip the first instruction, since we always process the previous
        // instruction
        if(ins == m.begin())
            continue;
        const auto i = std::prev(ins);

        if(i->name() == "identity")
        {
            m.replace_instruction(i, i->inputs().front());
            m.move_instruction(i, m.end());
        }
        if(ins == last)
        {
            if(ins->name() == "identity")
            {
                const instruction_ref& identity_input = ins->inputs().front();
                if(identity_input->outputs().size() == 1)
                {
                    m.move_instruction(identity_input, i);
                    // since this is the last instruction, removing it only
                    // requires changing "last" and calling remove below
                    last = std::prev(last);
                }
            }
            break;
        }
    }
    m.remove_instructions(std::next(last), m.end());
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
