#include <migraphx/common_subexpression_elimination.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/functional.hpp>

#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

template <class Range>
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
        for(const auto& pp : found_instructions)
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

void common_subexpression_elimination::apply(program& p) const { cse_range(p, iterator_for(p)); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
