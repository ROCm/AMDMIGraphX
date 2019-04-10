#include <migraphx/program.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/fp_conversion.hpp>
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
    assert(ins->get_shape().type() == shape::float_type ||
           ins->get_shape().type() == shape::double_type);
    assert(contains({"@literal", "@param"}, ins->name()));
    instruction_ref ins_fp16{};
    if(ins->name() == "@literal")
    {
        shape s = ins->get_shape();
        auto l  = ins->get_literal();
        if(s.type() == shape::float_type)
        {
            auto tv = l.get<const float>();
            ins_fp16 =
                prog.add_literal(literal({shape::half_type, s.lens()}, tv.begin(), tv.end()));
        }
        else
        {
            auto tv = l.get<const double>();
            ins_fp16 =
                prog.add_literal(literal({shape::half_type, s.lens()}, tv.begin(), tv.end()));
        }
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
    bool reduced_precision  = false;
    shape::type_t orig_type = shape::float_type;
    for(auto ins : iterator_for(prog))
    {
        // convert float_type to half_type
        if(contains({"@literal", "@param"}, ins->name()) &&
           (ins->get_shape().type() == shape::float_type ||
            ins->get_shape().type() == shape::double_type))
        {
            orig_type     = ins->get_shape().type();
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
            if(!contains({"@literal", "@param"}, ins->name()))
            {
                ins->recompute_ins_shape();
            }
        }

        auto ins = std::prev(prog.end());
        if(ins->get_shape().type() == shape::half_type)
        {
            prog.add_instruction(op::fp_conversion{orig_type}, ins);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
