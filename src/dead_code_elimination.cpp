#include <migraph/dead_code_elimination.hpp>
#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/functional.hpp>

namespace migraph {

void dead_code_elimination::apply(program& p) const
{
    for(auto i:iterator_for(p)) 
    {
        // Skip over instructions that may have been removed
        if(!p.has_instruction(i))
            continue;
        // Skip the last instruction
        if(i == std::prev(p.end()))
            break;
        fix([&](auto self, auto ins) {
            assert(p.has_instruction(ins));
            if(ins->output.empty())
            {
                std::cout << p << std::endl;
                auto args = ins->arguments;
                p.remove_instruction(ins);
                for(auto arg:args)
                    self(arg);
            }
        })(i);
    }
}

} // namespace migraph
