#include <migraphx/program.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/target.hpp>
#include <migraphx/env.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/time.hpp>
#include <migraphx/iterator_for.hpp>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

instruction_ref convert_fp32_fp16(program& prog, instruction_ref& ins)
{
    assert(ins->get_shape().type() == shape::float_type);
    assert(ins->name().front() == '@');
    instruction_ref ins_fp16{};
    if(ins->name() == "@literal")
    {
        std::vector<float> values;
        auto l_fp32 = ins->get_literal();
        shape s     = ins->get_shape();
        l_fp32.visit([&](auto val) { values.assign(val.begin(), val.end()); });
        ins_fp16 = prog.add_literal(literal({shape::half_type, s.lens()}, values));
    }
    else if(ins->name() == "@param")
    {
        if(ins == std::prev(prog.end()))
        {
            ins_fp16 = prog.add_instruction(op::fp_conversion{}, ins);
        }
        else
        {
            ins_fp16 = prog.insert_instruction(std::next(ins), op::fp_conversion{}, ins);
        }
    }

    return ins_fp16;
}

void quantize(program& prog)
{
    bool reduced_precision = false;
    for(auto ins : iterator_for(prog))
    {
        // convert float_type to half_type
        if(ins->name().front() == '@' && ins->get_shape().type() == shape::float_type)
        {
            auto ins_fp16 = convert_fp32_fp16(prog, ins);
            auto outputs  = ins->outputs();
            for(auto output : outputs)
            {
                if(output != ins_fp16)
                {
                    instruction::replace_argument(output, ins, ins_fp16, false);
                }
            }

            reduced_precision = true;
        }
    }

    // add another instruction at last to convert fp16 to fp32
    if(reduced_precision)
    {
        for(auto ins : iterator_for(prog))
        {
            if(ins->name().front() != '@')
            {
                ins->recompute_ins_shape();
            }
        }

        auto ins = std::prev(prog.end());
        if(ins->get_shape().type() == shape::half_type)
        {
            prog.add_instruction(op::fp_conversion{false}, ins);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
