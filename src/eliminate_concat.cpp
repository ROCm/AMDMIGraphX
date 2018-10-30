#include <iterator>
#include <migraph/eliminate_concat.hpp>
#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/dfor.hpp>

namespace migraph {
void eliminate_concat::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        // Look for the concat operator
        if(ins->name() != concat_opt.name())
            continue;
        // If any inputs are literals then abort
        if(std::any_of(ins->inputs().begin() + 1, ins->inputs().end(), [](auto arg) {
               return arg->name() == "@literal";
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

            for(auto ins2 = ins->inputs().begin(); ins2 != ins->inputs().end() - 1; ins2++)
            {
                auto last2 = (*ins2)->inputs().back();
                if(last2->name() == concat_opt.allocate())
                {
                    allocations.push_back(last2);
                }
            }
            // Need to sort the allocations, so that we know where to
            // insert the "super"-allocation
            std::sort(
                allocations.begin(), allocations.end(), [&](instruction_ref x, instruction_ref y) {
                    return std::distance(p.begin(), x) < std::distance(p.begin(), y);
                });
            // Move "super" allocation to the front
            auto first         = allocations.front();
            auto super         = p.move_instruction(last, first);
            std::size_t offset = 0;
            for(auto x : allocations)
            {
                migraph::op::load op{x->get_shape(), offset};
                p.replace_instruction(x, op, {super});
                offset += x->get_shape().elements();
            }
            std::vector<instruction_ref> args = {super};
            std::copy(ins->inputs().begin(), ins->inputs().end() - 1, std::back_inserter(args));
            p.replace_instruction(ins, migraph::op::identity{}, args);
        }
    }
}
} // namespace migraph
