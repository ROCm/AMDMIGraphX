#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/op/contiguous.hpp>
#include <migraphx/op/identity.hpp>
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

        // if no changes for the shape, the contiguous can also be removed
        if(new_shape == ins->get_shape())
        {
            return true;
        }

        auto outputs = ins->outputs();
        // If the current instruction has no output, it means it is the last
        // instruction and generates a non-standard output shape, and the last
        // output shape is different from the case with the contiguous operator
        if(outputs.empty())
        {
            return false;
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
        // return instruction should have inputs with standard shape
        if (ins->name() == "return")
            continue;
            
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
                else if(prev->can_eval())
                {
                    auto c = op::contiguous{};
                    auto r = c.compute(c.compute_shape({prev->get_shape()}), {prev->eval()});

                    auto l = p.add_literal(r.get_shape(), r.data());
                    p.replace_instruction(arg, l);
                }
            }
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
