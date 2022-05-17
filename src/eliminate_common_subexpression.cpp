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
void cse_range(module& m, Range&& r)
{
    std::unordered_multimap<std::string, instruction_ref> instructions;
    std::unordered_set<instruction_ref> processed_ins;
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
            if(contains(processed_ins, eq))
                continue;
            if(*eq != *ins)
                continue;
            m.replace_instruction(ins, eq);
            processed_ins.emplace(ins);
            std::vector<instruction_ref> outputs;
            std::copy_if(eq->outputs().begin(),
                         eq->outputs().end(),
                         std::back_inserter(outputs),
                         [&](auto x) { return m.has_instruction(x); });

            std::sort(outputs.begin(), outputs.end(), [&](auto x, auto y) {
                return std::distance(eq, x) < std::distance(eq, y);
            });
            cse_range(m, outputs);
        }
        instructions.emplace(ins->name(), ins);
    }
}

void eliminate_common_subexpression::apply(module& m) const { cse_range(m, iterator_for(m)); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
