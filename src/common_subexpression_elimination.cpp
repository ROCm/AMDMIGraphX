#include <migraph/common_subexpression_elimination.hpp>
#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/ranges.hpp>
#include <migraph/functional.hpp>

namespace migraph {

template<class Range>
void cse_range(program& p, Range&& r)
{
    std::unordered_multimap<std::string, instruction_ref> instructions;
    for(auto ins : r)
    {
        // Skip dead instructions
        if(ins->outputs().empty())
            continue;
        // Find instruction with the same name
        auto found_instructions = range(instructions.equal_range(ins->name()));
        for(auto pp:found_instructions)
        {
            auto eq = pp.second;
            if(*eq != *ins)
                continue;
            p.replace_instruction(ins, eq);
            cse_range(p, eq->outputs());
        }
        instructions.emplace(ins->name(), ins);
    }
}

void common_subexpression_elimination::apply(program& p) const
{
    cse_range(p, iterator_for(p));
}

} // namespace migraph
