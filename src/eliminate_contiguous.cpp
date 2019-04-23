#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool try_compute_shape(instruction_ref ins, const std::vector<shape>& inputs)
{
    try
    {
        shape new_shape = ins->get_operator().compute_shape(inputs);
        // If the output shape is a standard shape, no need to try its output 
        if (new_shape.standard())
        {
            return true;
        }

        auto outputs = ins->outputs();
        // If the current instruction has no output, it means the last output shape
        // is non-standard, then we cannot eliminate the contiguous
        if (outputs.empty())
        {
            return false;
        }

        for (auto output : outputs)
        {
            auto args = output->inputs();
            std::vector<shape> input_shapes;
            for (auto arg : args)
            {
                input_shapes.push_back((arg == ins) ? new_shape : arg->get_shape());
            }
            
            if (!try_compute_shape(output, input_shapes))
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

bool try_compute_shape(instruction_ref ins, const std::vector<instruction_ref>& args)
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
