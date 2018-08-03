#include <migraph/auto_contiguous.hpp>
#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/iterator_for.hpp>

namespace migraph {

void auto_contiguous::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        shape s = ins->result;
        if(not s.standard())
        {
            auto prev = p.insert_instruction(ins, ins->op, ins->arguments);
            p.replace_instruction(ins, contiguous{}, prev);
        }
    }
}

} // namespace migraph
