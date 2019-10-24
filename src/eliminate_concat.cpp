#include <iterator>
#include <migraphx/eliminate_concat.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/load.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/dfor.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
void eliminate_concat::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        // Look for the concat operator
        if(ins->name() != concat_opt.name())
            continue;
        // If any inputs are builtin or context free then abort
        // If any inputs are used more than once, then abort since there could
        // be errors due to aliasing
        if(std::any_of(ins->inputs().begin(), ins->inputs().end(), [](auto arg) {
               return arg->name().front() == '@' or arg->get_operator().is_context_free() or
                      arg->outputs().size() > 1;
           }))
            continue;
        // We can only do this optimization when concat axis is either the leftmost
        // axis OR the sizes to the left of this axis are all equal to 1
        // Since we've already checked that the non-axis dimensions are identical
        // we only need to check the first input
        auto lens      = ins->inputs().front()->get_shape().lens();
        auto concat_op = concat_opt.get_concat(ins->get_operator());
        if(concat_op.axis == 0 ||
           std::all_of(lens.begin(), lens.begin() + concat_op.axis, [](auto x) { return x == 1; }))
        {
            // Last input should be an allocation
            auto last = ins->inputs().back();
            if(last->name() != concat_opt.allocate())
                continue;
            // Where are the allocations for the tensors to be concatenated?
            std::vector<instruction_ref> allocations;

            std::transform(
                ins->inputs().begin(),
                std::prev(ins->inputs().end()),
                std::back_inserter(allocations),
                [&](instruction_ref x) { return instruction::get_output_alias(x, true); });

            if(std::any_of(allocations.begin(), allocations.end(), [&](auto x) {
                   return x->name() != concat_opt.allocate();
               }))
                continue;

            // Need to sort the allocations, so that we know where to
            // insert the "super"-allocation
            std::sort(
                allocations.begin(), allocations.end(), [&](instruction_ref x, instruction_ref y) {
                    return std::distance(p.begin(), x) < std::distance(p.begin(), y);
                });
            // Move "super" allocation to the front
            auto first = allocations.front();
            auto super = p.move_instruction(last, first);
            // Replace each allocation with a load
            std::size_t offset = 0;
            for(auto alloc : allocations)
            {
                op::load op{alloc->get_shape(), offset};
                p.replace_instruction(alloc, op, {super});
                offset += alloc->get_shape().bytes();
            }
            std::vector<instruction_ref> args = {super};
            std::copy(ins->inputs().begin(), ins->inputs().end() - 1, std::back_inserter(args));
            p.replace_instruction(ins, migraphx::op::identity{}, args);
        }
    }
}
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
