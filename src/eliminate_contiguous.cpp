#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

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
        // skip the reshape operator for now, since there is a bug
        // for the transpose followed by a reshape
        if(ins->name() == "reshape")
        {
            continue;
        }

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

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
