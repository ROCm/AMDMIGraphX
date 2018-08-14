#include <migraph/eliminate_contiguous.hpp>
#include <migraph/program.hpp>
#include <migraph/instruction.hpp>
#include <migraph/operators.hpp>
#include <migraph/iterator_for.hpp>
#include <migraph/ranges.hpp>
#include <migraph/stringutils.hpp>

namespace migraph {

bool try_compute_shape(operation op, std::vector<instruction_ref> args)
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
        auto args = ins->arguments;
        for(auto arg : ins->arguments)
        {
            // TODO: Pass in names for the operator in the constructor instead
            // of using ends_with
            if(ends_with(arg->op.name(), "contiguous"))
            {
                auto new_args = args;
                auto prev = arg->arguments.front();
                replace(new_args, arg, prev);
                if(try_compute_shape(ins->op, new_args)) 
                {
                    replace_argument(ins, arg, prev);
                }
            }
        }
    }
}

} // namespace migraph
