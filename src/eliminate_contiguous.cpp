#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static bool try_compute_shape(instruction_ref ins, const std::vector<shape>& inputs)
{
    try
    {
        shape new_shape = ins->get_operator().compute_shape(inputs);
        // If the output shape is a standard shape, no need to try its output
        if(new_shape.standard())
        {
            return true;
        }

        auto outputs = ins->outputs();
        // If the current instruction has no output, it means it is the last
        // instruction and generates a non-standard output. But for unary
        // and binary operators, we can still remove it and reshape the output
        // to be standard since these operator can handle non-standard inputs
        if(outputs.empty())
        {
            return true;
        }

        for(auto output : outputs)
        {
            auto args = output->inputs();
            std::vector<shape> input_shapes(args.size());
            std::transform(args.begin(), args.end(), input_shapes.begin(), [&](auto& arg) {
                return (arg == ins) ? new_shape : arg->get_shape();
            });

            if(!try_compute_shape(output, input_shapes))
            {
                return false;
            }
        }
    }
    catch(...)
    {
        return false;
    }

    return true;
}

static bool try_compute_shape(instruction_ref ins, const std::vector<instruction_ref>& args)
{
    auto inputs = to_shapes(args);
    return try_compute_shape(ins, inputs);
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
                if(try_compute_shape(ins, new_args))
                {
                    instruction::replace_argument(ins, arg, prev);
                }
            }
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
