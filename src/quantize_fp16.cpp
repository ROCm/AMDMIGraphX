#include <migraphx/float_equal.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/quantize_fp16.hpp>
#include <migraphx/quantize_util.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/target.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static void convert_outputs_fp16(instruction_ref ins)
{
    auto inputs = ins->inputs();
    for(auto in : inputs)
    {
        if(in->get_shape().type() == shape::half_type)
        {
            continue;
        }

        if(in->get_shape().type() == shape::float_type or
           in->get_shape().type() == shape::double_type)
        {
            assert(in->name() == "convert" and
                   in->inputs().front()->get_shape().type() == shape::half_type);
            instruction::replace_argument(ins, in, in->inputs().front());
        }
    }
}

static void quantize_module(module& m,
                            const std::vector<std::string>& ins_names,
                            std::unordered_map<instruction_ref, instruction_ref>& map_fp16,
                            bool quantize_inout = false)
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "@return")
        {
            if(quantize_inout)
            {
                convert_outputs_fp16(ins);
            }
            break;
        }

        if(ins->name() == "@param" or ins->name() == "@literal")
        {
            auto s = ins->get_shape();
            if(s.type() == shape::float_type or s.type() == shape::double_type)
            {
                auto fp16_ins = m.insert_instruction(
                    std::next(ins), make_op("convert", {{"target_type", shape::half_type}}), ins);
                map_fp16[ins] = fp16_ins;
            }

            continue;
        }

        if(ins->name() == "convert" and ins->get_shape().type() == shape::half_type)
        {
            continue;
        }

        // all indicates every instruction is converted
        if((not contains(ins_names, "all")) and (not contains(ins_names, ins->name())))
        {
            continue;
        }

        shape orig_shape = ins->get_shape();
        // process all inputs, if input is a fp32 or fp64, convert it
        // to a fp16 by adding a convert operator.
        auto inputs = ins->inputs();
        std::vector<instruction_ref> converted_inputs;
        for(auto input : inputs)
        {
            auto s = input->get_shape();
            if(s.type() != shape::float_type and s.type() != shape::double_type)
            {
                converted_inputs.push_back(input);
                continue;
            }

            // if the input is a parameter of a subgraph
            instruction_ref input_fp16{};
            if(input->name() == "@param")
            {
                auto in_outs = input->outputs();
                auto it      = std::find_if(in_outs.begin(), in_outs.end(), [](auto o) {
                    return (o->name() == "convert" and o->get_shape().type() == shape::half_type);
                });
                assert(it != in_outs.end());
                input_fp16 = *it;
                converted_inputs.push_back(input_fp16);
            }
            // if the input is a convert operator, uses its input
            // as its current input
            else if(input->name() == "convert" and
                    input->inputs().front()->get_shape().type() == shape::half_type)
            {
                input_fp16 = input->inputs().front();
                converted_inputs.push_back(input_fp16);
            }
            else
            {
                input_fp16 = insert_quant_ins(m, ins, input, shape::half_type, map_fp16);
                converted_inputs.push_back(input_fp16);
            }
        }

        auto mod_inputs = ins->module_inputs();
        for(auto*& smod : mod_inputs)
        {
            quantize_module(*smod, ins_names, map_fp16, true);
        }

        auto op        = ins->get_operator();
        auto ins_shape = compute_shape(op, converted_inputs, mod_inputs);
        if(ins_shape != orig_shape)
        {
            // tuple type, followed by get_tuple_elem
            if(ins_shape.type() == shape::tuple_type)
            {
                auto outputs = ins->outputs();
                for(auto out : outputs)
                {
                    auto out1 = m.insert_instruction(
                        std::next(out),
                        make_op("convert", {{"target_type", out->get_shape().type()}}),
                        out);
                    m.replace_instruction(out, out1);
                }
            }
            else
            {
                // check the dead code case to avoid assert
                auto ins_orig_shape = m.insert_instruction(
                    std::next(ins), make_op("convert", {{"target_type", orig_shape.type()}}), ins);
                m.replace_instruction(ins, ins_orig_shape);
            }
        }
        m.replace_instruction(ins, op, converted_inputs, mod_inputs);
    }
}

// This function is to convert any instructions specified in the input
// from double or float to float16 by inserting a convert operator.
// For the conversion, there could be cases of overflowing, but it
// is very rare in the area of deeping learning, so we just do a
// truncate of the input to get the fp16.
void quantize_fp16_pass::apply(program& prog) const
{
    auto* mm = prog.get_main_module();
    std::unordered_map<instruction_ref, instruction_ref> map_fp16;
    quantize_module(*mm, ins_names, map_fp16, false);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
