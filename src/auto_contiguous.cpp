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
        shape s = ins->get_shape();
        if(not s.standard())
        {
            auto c = p.insert_instruction(std::next(ins), contiguous{}, ins);
            p.replace_instruction(ins, c);
        }
    }
}

} // namespace migraph
