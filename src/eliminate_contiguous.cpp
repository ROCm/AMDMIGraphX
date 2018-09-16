#include <migraph/eliminate_contiguous.hpp>
#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/ranges.hpp>
#include <migraph/stringutils.hpp>
#include <utility>

namespace migraph {

bool try_compute_shape(const operation& op, const std::vector<instruction_ref>& args)
{
    try
    {
        compute_shape(op, args);
    }
    catch(...)
    {
        return false;
    }
    return true;
}

void eliminate_contiguous::apply(program& p) const
{
    for(auto ins : iterator_for(p))
    {
        // Make a copy so we can modify it while we iterate
        auto args = ins->inputs();
        for(auto arg : ins->inputs())
        {
            // TODO: Pass in names for the operator in the constructor instead
            // of using ends_with
            if(ends_with(arg->name(), "contiguous"))
            {
                auto new_args = args;
                auto prev     = arg->inputs().front();
                replace(new_args, arg, prev);
                if(try_compute_shape(ins->get_operator(), new_args))
                {
                    instruction::replace_argument(ins, arg, prev);
                }
            }
        }
    }
}

} // namespace migraph
