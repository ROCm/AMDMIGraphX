#include <migraphx/eliminate_common_subexpression.hpp>
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
            auto outputs = eq->outputs();
            std::sort(outputs.begin(), outputs.end(), [&](auto x, auto y) {
                return std::distance(eq, x) < std::distance(eq, y);
            });
            cse_range(p, outputs);
        }
        instructions.emplace(ins->name(), ins);
    }
}

void eliminate_common_subexpression::apply(program& p) const { cse_range(p, iterator_for(p)); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
