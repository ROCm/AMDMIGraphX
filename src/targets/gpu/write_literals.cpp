#include <migraph/gpu/write_literals.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/instruction.hpp>

namespace migraph {

namespace miopen {

void write_literals::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        if(ins->op.name() == "@literal")
        {
            literal l = ins->lit;
            auto pre  = p.add_literal(l);
            p.replace_instruction(ins, hip_write{}, pre);
        }
    }
}

} // namespace miopen

} // namespace migraph
