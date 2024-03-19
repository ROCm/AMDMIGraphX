#ifndef MIGRAPHX_GUARD_MIGRAPHX_LIVENESS_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_LIVENESS_HPP

#include <migraphx/config.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <unordered_set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

// This will do liveness analysis on the module, and it will call the
// function `f` with the instruction and the set of the other instructions
// that are live
template <class F>
void liveness(const module& m, F f)
{
    auto implicit_deps = m.calc_implicit_deps();
    std::unordered_set<instruction_ref> live_set;
    auto rp = reverse(m);
    for(auto rins : iterator_for(rp)) // NOLINT
    {
        // The base iterator is one ahead, so we need to use the previous iterator
        auto ins = std::prev(rins.base());
        // Add live variables
        auto add_live_variables = [&](const auto& inputs) {
            for(auto input : inputs)
            {
                auto i = instruction::get_output_alias(input);
                // Skip if variable comes from parent
                if(not m.has_instruction(i))
                    continue;
                live_set.insert(i);
            }
        };
        add_live_variables(ins->inputs());
        add_live_variables(implicit_deps[ins]);
        // Remove last usage
        auto it = live_set.find(ins);
        if(it != live_set.end())
        {
            live_set.erase(it);
            f(ins, live_set);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_LIVENESS_HPP
