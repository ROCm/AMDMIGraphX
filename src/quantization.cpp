#include <migraphx/quantization.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/convert.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/mul.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/quant_dot.hpp>
#include <migraphx/op/capture.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/quant_convolution.hpp>
#include <migraphx/op/multibroadcast.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/ranges.hpp>
#include <utility>
#include <iomanip>
#include <fstream>

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

    if(ins->name() == "undefined")
    {
        return ins;
    }

    if(scale < 0.0f)
    {
        MIGRAPHX_THROW("INSERT_QUANT_INS: scale less than 0");
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

static std::vector<std::pair<float, float>> int8_quant_params;

// function to compute the scale for each convert operator to convert to int8
void calc_quant_params(std::size_t ins_index, std::vector<migraphx::argument> args)
{
    std::pair<float, float> param_pair{1.0f, 0.0f};

    // scale and shift is need for only int8 type, and we do not
    // consider shift, so set shift to 0
    std::vector<float> vec_val;
    args.front().visit([&](auto output) { vec_val.assign(output.begin(), output.end()); });
    auto max_val     = *std::max_element(vec_val.begin(), vec_val.end());
    auto min_val     = *std::min_element(vec_val.begin(), vec_val.end());
    auto max_abs     = std::max(std::fabs(max_val), std::fabs(min_val));
    param_pair.first = 127.0f / max_abs;

    int8_quant_params[ins_index] = param_pair;
};

// int8 quantization is different from fp16 since int8 can only handle value
// -128 ~ 127. To convert the float or double to int8, we need a scale and
// a shift, then the convert can be done as v_int8 = fp * scale + shift.
// To simplify the changes, we consider shift as 0.0f for now.
void quantize_int8(program& prog,
                   const std::vector<std::string>& ins_names,
                   const std::vector<std::pair<float, float>>& quant_params)
{
    for(size_t i = 0; i < quant_params.size(); i++)
    {
        auto param = quant_params.at(i);
        std::cout << "index = " << i << ", scale = " << param.first << "\t" << param.second
                  << std::endl;
    }
    std::cout << std::endl;

    // For now, we only support the int8 quantization of gemm and convolution
    std::vector<std::string> op_names = {"dot", "convolution"};
    if(!std::all_of(ins_names.begin(), ins_names.end(), [&](auto name) {
           return (std::find(op_names.begin(), op_names.end(), name) != op_names.end());
       }))
    {
        MIGRAPHX_THROW("QUANTIZE_INT8: only support DOT and CONVOLUTION operation");
    }

    std::size_t quant_param_index = 0;
    std::unordered_map<instruction_ref, instruction_ref> map_quant_ins;
    std::unordered_map<instruction_ref, std::size_t> map_index;
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
        auto inputs = ins->inputs();
        std::vector<std::pair<float, float>> ins_quant_params;
        for(auto input : inputs)
        {
            // calculate the index of each instruction to be quantized
            if(map_index.count(input) == 0)
            {
                map_index[input] = quant_param_index++;
            }
            auto param = quant_params[map_index[input]];
            ins_quant_params.push_back(param);

            // In general, the target_type is int8, but for the dot
            // operation, if it has 3 inputs, then the last one should
            // be converted to int32_type
            shape::type_t quant_type = shape::int8_type;
            if(ins->name() == "dot" and inputs.size() == 3 and input == inputs.back())
            {
                quant_type = shape::int32_type;
            }

            auto s = input->get_shape();
            if((s.type() == shape::float_type || s.type() == shape::double_type ||
                s.type() == shape::int32_type) &&
               s.type() != quant_type)
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

        // When converting from other types to int8_type, there are parameters
        // used as scale and shift(.0f), which will generate results diffrent from
        // the original results. To adjust the output to be "correct(approximatly
        // equal)", we need additional calculation for the adjustment
        if(ins->name() == "dot")
        {
            auto dot_op = any_cast<op::dot>(ins->get_operator());
            float new_alpha =
                dot_op.alpha / (ins_quant_params[0].first * ins_quant_params[1].first);
            float new_beta = dot_op.beta;
            // We need additional checking about the quant_alpha value. If
            // abs(quant_alpha) > 50 (some tmp value set here), we can convert
            // it to an integer as the new_alpha in the quant_dot
            float threshold = 50.0f;
            if(fabs(new_alpha) >= threshold && fabs(new_beta) >= threshold)
            {
                int32_t quant_alpha = static_cast<int32_t>(new_alpha);
                int32_t quant_beta  = static_cast<int32_t>(new_beta);
                shape quant_shape   = compute_shape(op::quant_dot{1, 0}, converted_inputs);
                if(quant_shape.type() == orig_type)
                {
                    prog.replace_instruction(
                        ins, op::quant_dot{quant_alpha, quant_beta}, converted_inputs);
                }
                else
                {
                    auto quant_dot = prog.insert_instruction(
                        ins, op::quant_dot{quant_alpha, quant_beta}, converted_inputs);
                    prog.replace_instruction(ins, op::convert{orig_type}, quant_dot);
                }
            }
            // only alpha can be quantized, quantization of beta will cause
            // big error, so we have to manually do the multiplication and
            // addition
            else if(fabs(new_alpha) >= threshold)
            {
                // truncate to the nearest integer
                new_alpha           = new_alpha > 0.0 ? new_alpha + 0.5 : new_alpha - 0.5;
                int32_t quant_alpha = static_cast<int32_t>(new_alpha);
                int32_t quant_beta  = 0;
                if(orig_type == shape::int32_type)
                {
                    if(inputs.size() == 2 or dot_op.beta == 0.0f)
                    {
                        prog.replace_instruction(
                            ins, op::quant_dot{quant_alpha, quant_beta}, converted_inputs);
                    }
                    // if there are 3 inputs, we need to consider the third argument
                    else
                    {
                        auto q_dot = prog.insert_instruction(
                            ins, op::quant_dot{quant_alpha, quant_beta}, converted_inputs);
                        std::vector<float> vec_beta(q_dot->get_shape().elements(), dot_op.beta);
                        auto l_beta = prog.add_literal(literal{orig_type, vec_beta});
                        auto beta_c =
                            prog.insert_instruction(ins, op::mul{}, l_beta, inputs.back());
                        prog.replace_instruction(ins, op::add{}, q_dot, beta_c);
                    }
                }
                else
                {
                    if(inputs.size() == 2 or dot_op.beta == 0.0f)
                    {
                        auto q_dot = prog.insert_instruction(
                            ins, op::quant_dot{quant_alpha, quant_beta}, converted_inputs);
                        prog.replace_instruction(ins, op::convert{orig_type}, q_dot);
                    }
                    // if there are 3 inputs, we need to consider the third argument
                    else
                    {
                        auto q_dot = prog.insert_instruction(
                            ins, op::quant_dot{quant_alpha, quant_beta}, converted_inputs);
                        auto oq_dot = prog.insert_instruction(ins, op::convert{orig_type}, q_dot);
                        std::vector<float> vec_beta(q_dot->get_shape().elements(), dot_op.beta);
                        auto l_beta = prog.add_literal(literal{oq_dot->get_shape(), vec_beta});
                        auto beta_c =
                            prog.insert_instruction(ins, op::mul{}, l_beta, inputs.back());
                        prog.replace_instruction(ins, op::add{}, oq_dot, beta_c);
                    }
                }
            }
            else
            {
                auto q_dot = prog.insert_instruction(ins, op::quant_dot{1, 0}, converted_inputs);
                std::vector<float> vec_alpha(q_dot->get_shape().elements(), new_alpha);
                if(orig_type == shape::int32_type)
                {
                    auto l_alpha = prog.add_literal(literal(ins->get_shape(), vec_alpha));
                    if(converted_inputs.size() == 2 or dot_op.beta == 0.0f)
                    {
                        prog.replace_instruction(ins, op::mul{}, l_alpha, q_dot);
                    }
                    // case of 3 arguments
                    else
                    {
                        std::vector<float> vec_beta(ins->get_shape().elements(), new_beta);
                        auto l_beta   = prog.add_literal(literal(ins->get_shape(), vec_beta));
                        auto alpha_ab = prog.insert_instruction(ins, op::mul{}, l_alpha, q_dot);
                        auto beta_c =
                            prog.insert_instruction(ins, op::mul{}, l_beta, inputs.back());
                        prog.replace_instruction(ins, op::add{}, alpha_ab, beta_c);
                    }
                }
                else
                {
                    auto oq_dot  = prog.insert_instruction(ins, op::convert{orig_type}, q_dot);
                    auto l_alpha = prog.add_literal(literal(ins->get_shape(), vec_alpha));
                    if(converted_inputs.size() == 2 or dot_op.beta == 0.0f)
                    {
                        prog.replace_instruction(ins, op::mul{}, l_alpha, oq_dot);
                    }
                    // case of 3 arguments
                    else
                    {
                        std::vector<float> vec_beta(ins->get_shape().elements(), new_beta);
                        auto l_beta   = prog.add_literal(literal(ins->get_shape(), vec_beta));
                        auto alpha_ab = prog.insert_instruction(ins, op::mul{}, l_alpha, oq_dot);
                        auto beta_c =
                            prog.insert_instruction(ins, op::mul{}, l_beta, inputs.back());
                        prog.replace_instruction(ins, op::add{}, alpha_ab, beta_c);
                        // auto gemm_res = prog.insert_instruction(ins, op::add{}, alpha_ab,
                        // beta_c); prog.replace_instruction(ins, op::capture{0, print_gemm_res},
                        // gemm_res);
                    }
                }
            }
        }
        else if(ins->name() == "convolution")
        {
            // Current MIOpen convolution does not support alpha and beta,
            // so we need a separate multiply to adjust the output
            auto conv_op       = any_cast<op::convolution>(ins->get_operator());
            auto padding       = conv_op.padding;
            auto stride        = conv_op.stride;
            auto dilation      = conv_op.dilation;
            auto padding_mode  = conv_op.padding_mode;
            auto group         = conv_op.group;
            auto adjust_factor = 1.0 / (ins_quant_params[0].first * ins_quant_params[1].first);

            shape quant_shape =
                compute_shape(op::quant_convolution{padding, stride, dilation, padding_mode, group},
                              converted_inputs);
            std::vector<float> vec_factor(quant_shape.elements(), adjust_factor);
            auto fl = prog.add_literal(literal{{orig_type, quant_shape.lens()}, vec_factor});
            if(quant_shape.type() == orig_type)
            {
                if(adjust_factor == 1.0f)
                {
                    prog.replace_instruction(
                        ins,
                        op::quant_convolution{padding, stride, dilation, padding_mode, group},
                        converted_inputs);
                }
                else
                {
                    auto quant_conv = prog.insert_instruction(
                        ins,
                        op::quant_convolution{padding, stride, dilation, padding_mode, group},
                        converted_inputs);
                    prog.replace_instruction(ins, op::mul{}, quant_conv, fl);
                    // auto q_conv = prog.insert_instruction(ins, op::mul{}, quant_conv, fl);
                    // prog.replace_instruction(ins, op::capture{10000, print_conv_res}, q_conv);
                }
            }
            else
            {
                auto quant_conv = prog.insert_instruction(
                    ins,
                    op::quant_convolution{padding, stride, dilation, padding_mode, group},
                    converted_inputs);
                if(adjust_factor == 1.0f)
                {
                    prog.replace_instruction(ins, op::convert{orig_type}, quant_conv);
                }
                else
                {
                    auto oq_conv = prog.insert_instruction(ins, op::convert{orig_type}, quant_conv);
                    prog.replace_instruction(ins, op::mul{}, oq_conv, fl);
                }
            }
        }
        else
        {
            MIGRAPHX_THROW("QUANTIZE_INT8: does not support operator" + ins->name());
        }
    }

    if(quant_param_index != quant_params.size())
    {
        MIGRAPHX_THROW("QUANTIZE_INT8: number of scales does not match");
    }
}

void quantize_int8(program& prog, const std::vector<std::string>& ins_names)
{
    quantize_int8(prog, ins_names, int8_quant_params);
}

void quantize_int8(program& prog)
{
    std::vector<std::string> ins_names = {"dot", "convolution"};
    quantize_int8(prog, ins_names, int8_quant_params);
}

// For the input of each input argument, we need to insert a
// capture operator to compute the scale and shift
void capture_arguments(program& prog,
                       const std::vector<std::string>& ins_names,
                       std::function<void(std::size_t, std::vector<argument>)> func)
{
    size_t num_quant_params = 0;
    // the int8 quantization only support dot and convolution
    std::vector<std::string> op_names = {"dot", "convolution", "quant_dot", "quant_convolution"};
    if(!std::all_of(ins_names.begin(), ins_names.end(), [&](auto name) {
           return std::find(op_names.begin(), op_names.end(), name) != op_names.end();
       }))
    {
        MIGRAPHX_THROW("CAPTURE_ARGUMENTS: input operator is not supported");
    }

    std::unordered_map<instruction_ref, instruction_ref> ins_map;
    for(auto ins : iterator_for(prog))
    {
        if(not contains(ins_names, ins->name()))
        {
            continue;
        }

        auto inputs = ins->inputs();
        std::vector<instruction_ref> new_args;
        for(auto input : inputs)
        {
            instruction_ref new_ins{};
            if(ins_map.count(input) > 0)
            {
                new_ins = ins_map[input];
            }
            else
            {
                new_ins = prog.insert_instruction(
                    std::next(input), op::capture{num_quant_params++, func}, input);
                ins_map[input] = new_ins;
            }
            new_args.push_back(new_ins);
        }
        instruction::replace(ins, ins->get_operator(), ins->get_shape(), new_args);
    }

    // set one pair of parameter for each argument
    int8_quant_params.resize(num_quant_params, std::make_pair(-1.0f, -1.0f));
}

void capture_arguments(program& prog, const std::vector<std::string>& ins_names)
{
    capture_arguments(prog, ins_names, calc_quant_params);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
