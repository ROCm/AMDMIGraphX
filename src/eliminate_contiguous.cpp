/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/eliminate_contiguous.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/op/contiguous.hpp>
#include <migraphx/op/identity.hpp>
#include <migraphx/par_for.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static bool try_compute_shape(instruction_ref ins,
                              const std::vector<shape>& inputs,
                              const std::vector<module_ref>& mods)
{
    try
    {
        shape new_shape = ins->get_operator().compute_shape(inputs, mods);
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

            if(!try_compute_shape(output, input_shapes, mods))
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

static bool try_compute_shape(instruction_ref ins,
                              const std::vector<instruction_ref>& args,
                              const std::vector<module_ref>& mods)
{
    auto inputs = to_shapes(args);
    return try_compute_shape(ins, inputs, mods);
}

void eliminate_contiguous::apply(module& m) const
{
    std::vector<instruction_ref> const_instruction;

    for(auto ins : iterator_for(m))
    {
        // return instruction should have inputs with standard shape
        if(ins->name() == "@return")
            continue;

        // Make a copy so we can modify it while we iterate
        auto args     = ins->inputs();
        auto new_args = args;
        auto mod_args = ins->module_inputs();

        for(auto arg : ins->inputs())
        {
            if(arg->name() == op_name)
            {
                auto prev = arg->inputs().front();
                replace(new_args, arg, prev);
                if(try_compute_shape(ins, new_args, mod_args))
                {
                    instruction::replace_argument(ins, arg, prev);
                }
                else if(prev->can_eval())
                {
                    const_instruction.push_back(arg);
                }
            }
        }
    }

    // Perform evaluations in parallel
    std::vector<argument> literals(const_instruction.size());
    par_for(const_instruction.size(), 1, [&](const auto i) {
        auto c      = op::contiguous{};
        auto prev   = const_instruction[i]->inputs().front();
        literals[i] = c.compute(c.compute_shape({prev->get_shape()}), {prev->eval()});
    });

    for(size_t i = 0; i < const_instruction.size(); i++)
    {
        auto l = m.add_literal(literals[i].get_shape(), literals[i].data());
        m.replace_instruction(const_instruction[i], l);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
