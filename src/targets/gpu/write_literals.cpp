#include <migraph/gpu/write_literals.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/instruction.hpp>

namespace migraph {

namespace gpu {

void write_literals::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
#if 0        
        if(ins->op.name() == "@literal")
        {
            literal l = ins->lit;
            auto pre  = p.add_literal(l);
            p.replace_instruction(ins, hip_write{}, pre);
        }
#else
        if (ins->op.name() == "write_literal") {
            p.replace_instruction(ins, hip_memcpy{}, ins->arguments);
        }
#endif        
    }
}

} // namespace gpu

} // namespace migraph
