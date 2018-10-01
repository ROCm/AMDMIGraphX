#include <migraph/gpu/lowering_mem_coloring.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/gpu/hip.hpp>
#include <migraph/instruction.hpp>

namespace migraph {

namespace gpu {

void lowering_mem_coloring::apply(program& p) const
{
    assert(ctx != nullptr);
    for(auto ins : iterator_for(p))
    {
        if(ins->op.name() == "write_literal")
        {
            std::vector<instruction_ref>& args = ins->arguments;
            const std::size_t* p_data = reinterpret_cast<const std::size_t*>(args.at(1)->lit.data());
            std::size_t offset = p_data[0];
            p.replace_instruction(ins, hip_memcpy{offset}, {args.at(0), args.at(2)});
            p.remove_instruction(args.at(1));
        }
    }
}
} // namespace gpu
} // namespace migraph
