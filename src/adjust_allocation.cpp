#include <migraphx/adjust_allocation.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/iterator_for.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void adjust_allocation::apply(module& p) const
{
    for(auto ins : iterator_for(p))
    {
        // skip instruction with no input
        if(ins->inputs().empty())
            continue;

        // Skip target-independent operators
        if(ins->get_operator().is_context_free())
            continue;

        auto alias_ins = instruction::get_output_alias(ins, true);
        if(alias_ins->name() != model.name())
            continue;
        // shape allocated is different from actual shape
        // of the instruction, reallocate and replace the previous one
        if(alias_ins->get_shape() == ins->get_shape())
            continue;
        auto alloc_ins = p.insert_instruction(ins, model.allocate(ins->get_shape()));
        p.replace_instruction(alias_ins, alloc_ins);

        
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
