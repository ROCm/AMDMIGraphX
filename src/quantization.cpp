#include <migraphx/quantization.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/convert.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/quant_dot.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/quant_convolution.hpp>
#include <migraphx/op/multibroadcast.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/ranges.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

instruction_ref insert_quant_ins(program& prog,
                                 instruction_ref& ins,
                                 shape::type_t type,
                                 std::unordered_map<instruction_ref, instruction_ref>& map_ins,
                                 float scale = 1.0f,
                                 float shift = 0.0f)
{
    if(map_ins.count(ins) > 0)
    {
        return map_ins[ins];
    }

    assert(ins->get_shape().type() == shape::float_type ||
           ins->get_shape().type() == shape::double_type ||
           ins->get_shape().type() == shape::int32_type);
    instruction_ref quant_ins{};
    quant_ins    = prog.insert_instruction(std::next(ins), op::convert{type, scale, shift}, ins);
    map_ins[ins] = quant_ins;

    return quant_ins;
}

// This function is to convert any instructions specified in the input
// from double or float to float16 by inserting a convert operator.
// For the conversion, there could be cases of overflowing, but it
// is very rare in the area of deeping learning, so we just do a
// truncate of the input to get the fp16.
void quantize(program& prog, const std::vector<std::string>& ins_names)
{
    std::unordered_map<instruction_ref, instruction_ref> map_fp16;
    for(auto ins : iterator_for(prog))
    {
        // all indicates every instruction is converted
        if((not contains(ins_names, "all")) and (not contains(ins_names, ins->name())))
        {
            continue;
        }

        shape::type_t orig_type = ins->get_shape().type();
        // process all inputs, if input is a fp32 or fp64, convert it
        // to a fp16 by adding a convert operator.
        auto inputs = ins->inputs();
        std::vector<instruction_ref> converted_inputs;
        for(auto input : inputs)
        {
            auto s = input->get_shape();
            if(s.type() == shape::float_type || s.type() == shape::double_type)
            {
                // if the input is a convert operator, uses its input
                // as its current input
                instruction_ref input_fp16{};
                if(input->name() == "convert")
                {
                    input_fp16 = input->inputs().front();
                }
                else
                {
                    input_fp16 = insert_quant_ins(prog, input, shape::half_type, map_fp16);
                }
                converted_inputs.push_back(input_fp16);
            }
            else
            {
                converted_inputs.push_back(input);
            }
        }

        // no change for the input, go to the next instruction
        if(inputs == converted_inputs)
        {
            continue;
        }

        auto op        = ins->get_operator();
        auto ins_shape = compute_shape(op, converted_inputs);
        if(ins_shape.type() != orig_type)
        {
            // check the dead code case to avoid assert
            bool output_empty = ins->outputs().empty();
            auto ins_orig_type =
                prog.insert_instruction(std::next(ins), op::convert{orig_type}, ins);
            if(!output_empty)
            {
                prog.replace_instruction(ins, ins_orig_type);
            }
        }

        prog.replace_instruction(ins, op, converted_inputs);
    }
}

void quantize(program& prog) { quantize(prog, {"all"}); }

// int8 quantization is different from fp16 since int8 can only handle value
// -128 ~ 127. To convert the float or double to int8, we need a scale and
// a shift, then the convert can be done as v_int8 = fp * scale + shift.
// To simplify the changes, we consider shift as 0.0f for now.
void quantize_int8(program& prog, const std::vector<std::string>& ins_names)
{
    // For now, we only support the int8 quantization of gemm and convolution
    std::vector<std::string> op_names = {"dot", "convolution"};
    if(!std::all_of(ins_names.begin(), ins_names.end(), [&](auto name) {
           return (std::find(op_names.begin(), op_names.end(), name) != op_names.end());
       }))
    {
        MIGRAPHX_THROW("QUANTIZE_INT8: only support DOT and CONVOLUTION operation");
    }

    // tmp value used just testing
    std::vector<std::pair<float, float>> int8_param{{1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f}};

    std::unordered_map<instruction_ref, instruction_ref> map_quant_ins;
    for(auto ins : iterator_for(prog))
    {
        if(not contains(ins_names, ins->name()))
        {
            continue;
        }

        shape::type_t orig_type = ins->get_shape().type();

        // for the dot operator, there could be 2 or 3 input arguments
        // if the 3rd argument is available, convert it to an int32.
        std::vector<instruction_ref> converted_inputs;

        // process all inputs, if input is a fp32 or fp64, convert it
        // to a int8 type by adding a convert operator and replace
        // the operator with the corresponding int8 version
        auto inputs             = ins->inputs();
        std::size_t param_index = 0;
        for(auto input : inputs)
        {
            // In general, the target_type is int8, but for the dot
            // operation, if it has 3 inputs, then the last one should
            // be converted to int32_type
            shape::type_t quant_type = shape::int8_type;
            if(ins->name() == "dot" and inputs.size() == 3 and input == inputs.back())
            {
                quant_type = shape::int32_type;
            }

            auto param = int8_param[param_index++];
            auto s     = input->get_shape();
            if(s.type() == shape::float_type || s.type() == shape::double_type ||
               s.type() == shape::int32_type)
            {
                // if the input is a convert operator, uses its input
                // as its current input
                instruction_ref quant_input{};
                if(input->name() == "convert")
                {
                    auto tmp_ins = input->inputs().front();
                    if(tmp_ins->get_shape().type() == quant_type)
                    {
                        quant_input = input->inputs().front();
                    }
                    else
                    {
                        quant_input = insert_quant_ins(
                            prog, input, quant_type, map_quant_ins, param.first, param.second);
                    }
                }
                else
                {
                    quant_input = insert_quant_ins(
                        prog, input, quant_type, map_quant_ins, param.first, param.second);
                }
                converted_inputs.push_back(quant_input);
            }
            else
            {
                converted_inputs.push_back(input);
            }
        }

        // no change for the input, go to the next instruction
        if(inputs == converted_inputs)
        {
            continue;
        }

        auto op        = ins->get_operator();
        auto ins_shape = compute_shape(op, converted_inputs);
        if(ins_shape.type() != orig_type)
        {
            // check the dead code case to avoid assert
            bool output_empty = ins->outputs().empty();
            // this conversion can be only from int32 to float or double
            auto ins_orig_type =
                prog.insert_instruction(std::next(ins), op::convert{orig_type}, ins);
            if(!output_empty)
            {
                prog.replace_instruction(ins, ins_orig_type);
            }
        }

        // When converting from other types to int8_type, there are parameters
        // used as scale and shift(.0f), which will generate results diffrent from
        // the original results. To adjust the output to be "correct(approximatly
        // equal)", we need additional calculation for the adjustment
        if (ins->name() == "dot")
        {
            auto dot_op = any_cast<op::dot>(ins->get_operator());
            int32_t quant_alpha = static_cast<int32_t>(dot_op.alpha / (int8_param[0].first * int8_param[1].first) + 0.5f);
            int32_t quant_beta = static_cast<int32_t>(dot_op.beta + 0.5f);
            prog.replace_instruction(ins, op::quant_dot{quant_alpha, quant_beta}, converted_inputs);
        }
        else if (ins->name() == "convolution")
        {
            // Current MIOpen convolution does not support alpha and beta, 
            // so we need a separate multiply to adjust the output
            auto conv_op = any_cast<op::convolution>(ins->get_operator());
            auto padding  = conv_op.padding;
            auto stride   = conv_op.stride;
            auto dilation = conv_op.dilation;
            auto padding_mode = conv_op.padding_mode;
            auto group = conv_op.group;
            auto adjust_factor = 1.0 / int8_param[0].first * int8_param[1].first;

            auto conv_res = prog.insert_instruction(ins, op::quant_convolution{padding, stride, dilation, padding_mode, group}, converted_inputs);
            auto conv_lens = conv_res->get_shape().lens();
            auto fl = prog.add_literal(literal(adjust_factor));
            auto adj_fact = prog.insert_instruction(ins, op::multibroadcast{conv_lens}, fl);
            prog.replace_instruction(ins, adj_fact);
        }
        else
        {
            MIGRAPHX_THROW("INT8_QUANTIZE: does not support operator" + ins->name());
        }

        prog.replace_instruction(ins, op, converted_inputs);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
