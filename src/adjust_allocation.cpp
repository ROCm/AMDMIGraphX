#include <migraphx/adjust_allocation.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>

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
        if(alias_ins->name() != model.name() and alias_ins->name() != "@param")
            continue;
        // shape allocated is different from actual shape
        // of the instruction, reallocate and replace the previous one
        if(alias_ins->get_shape() == ins->get_shape())
            continue;
        auto alloc_ins = p.insert_instruction(ins, model.allocate(ins->get_shape()));
        p.replace_instruction(alias_ins, alloc_ins);
        // If the memory is an output parameter then copy the memory to the parameter
        if(alias_ins->name() == "@param")
        {
            auto copy = p.insert_instruction(std::next(ins), make_op(model.copy()), ins, alias_ins);
            auto tail = range(std::next(copy), p.end());
            for(auto i : iterator_for(tail))
            {
                if(contains(i->inputs(), ins))
                    instruction::replace_argument(i, ins, copy);
            }
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
