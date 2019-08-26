#include <migraphx/quantization.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/convert.hpp>
#include <migraphx/op/clip.hpp>
#include <migraphx/op/round.hpp>
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
#include <migraphx/target.hpp>
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

    assert(ins->get_shape().type() == shape::float_type or
           ins->get_shape().type() == shape::double_type or
           ins->get_shape().type() == shape::int32_type);
    instruction_ref quant_ins{};
    auto insert_loc = std::next(ins);
    if(type == shape::int8_type)
    {
        auto scaled_ins = ins;
        if(scale != 1.0f)
        {
            auto float_ins = scaled_ins;
            if(scaled_ins->get_shape().type() != shape::float_type)
            {
                float_ins =
                    prog.insert_instruction(insert_loc, op::convert{shape::float_type}, scaled_ins);
            }
            std::vector<float> vec_scale(scaled_ins->get_shape().elements(), scale);
            auto l_scale = prog.add_literal(literal(scaled_ins->get_shape(), vec_scale));
            scaled_ins   = prog.insert_instruction(insert_loc, op::mul{}, l_scale, float_ins);
        }

        auto shifted_ins = scaled_ins;
        if(shift != 0.0f)
        {
            auto float_ins = shifted_ins;
            if(shifted_ins->get_shape().type() != shape::float_type)
            {
                float_ins = prog.insert_instruction(
                    insert_loc, op::convert{shape::float_type}, shifted_ins);
            }
            std::vector<float> vec_shift(shifted_ins->get_shape().elements(), shift);
            auto l_shift = prog.add_literal(literal(shifted_ins->get_shape(), vec_shift));
            shifted_ins  = prog.insert_instruction(insert_loc, op::add{}, l_shift, float_ins);
        }

        auto rounded_ins = prog.insert_instruction(insert_loc, op::round{}, shifted_ins);
        auto clipped_ins =
            prog.insert_instruction(insert_loc, op::clip{127.0f, -128.0f}, rounded_ins);
        quant_ins = prog.insert_instruction(insert_loc, op::convert{type}, clipped_ins);
    }
    else
    {
        quant_ins = prog.insert_instruction(insert_loc, op::convert{type}, ins);
    }

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
                if(input->name() == "convert" and
                   input->inputs().front()->get_shape().type() == shape::half_type)
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
            if((s.type() == shape::float_type or s.type() == shape::double_type or
                s.type() == shape::int32_type) and
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
                if(shape::int32_type == orig_type)
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
            // either alpha or beta cannot be quantized because of too big
            // relative rounding error
            else
            {
                if(converted_inputs.size() == 3)
                {
                    converted_inputs.pop_back();
                }
                auto q_dot   = prog.insert_instruction(ins, op::quant_dot{1, 0}, converted_inputs);
                auto f_dot   = prog.insert_instruction(ins, op::convert{shape::float_type}, q_dot);
                auto c_shape = q_dot->get_shape();
                std::vector<float> vec_alpha(c_shape.elements(), new_alpha);
                auto l_alpha =
                    prog.add_literal(literal({shape::float_type, c_shape.lens()}, vec_alpha));

                if(inputs.size() == 3 and dot_op.beta != 0.0f)
                {
                    auto alpha_ab = prog.insert_instruction(ins, op::mul{}, l_alpha, f_dot);
                    std::vector<float> vec_beta(c_shape.elements(), dot_op.beta);
                    auto l_beta =
                        prog.add_literal(literal({shape::float_type, c_shape.lens()}, vec_beta));
                    instruction_ref beta_c{};
                    if(orig_type != shape::float_type)
                    {
                        auto fp32_c = prog.insert_instruction(
                            ins, op::convert{shape::float_type}, inputs.back());
                        auto fp32_beta_c = prog.insert_instruction(ins, op::mul{}, l_beta, fp32_c);
                        beta_c = prog.insert_instruction(ins, op::convert{orig_type}, fp32_beta_c);
                    }
                    else
                    {
                        beta_c = prog.insert_instruction(ins, op::mul{}, l_beta, inputs.back());
                    }

                    if(orig_type == shape::float_type)
                    {
                        prog.replace_instruction(ins, op::add{}, alpha_ab, beta_c);
                    }
                    else
                    {
                        auto f_res = prog.insert_instruction(ins, op::add{}, alpha_ab, beta_c);
                        prog.replace_instruction(ins, op::convert{orig_type}, f_res);
                    }
                }
                else
                {
                    if(orig_type == shape::float_type)
                    {
                        prog.replace_instruction(ins, op::mul{}, l_alpha, f_dot);
                    }
                    else
                    {
                        auto alpha_ab = prog.insert_instruction(ins, op::mul{}, l_alpha, f_dot);
                        prog.replace_instruction(ins, op::convert{orig_type}, alpha_ab);
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
            auto adjust_factor = 1.0f / (ins_quant_params[0].first * ins_quant_params[1].first);

            auto quant_conv = prog.insert_instruction(
                ins,
                op::quant_convolution{padding, stride, dilation, padding_mode, group},
                converted_inputs);
            float threshold = 50.0f;
            std::vector<float> vec_factor(quant_conv->get_shape().elements(), adjust_factor);
            if(quant_conv->get_shape().type() == orig_type and adjust_factor >= threshold)
            {
                auto l_factor = prog.add_literal(
                    literal(quant_conv->get_shape(), vec_factor.begin(), vec_factor.end()));
                prog.replace_instruction(ins, op::mul{}, quant_conv, l_factor);
            }
            // convert quant_conv output to float type, multiply the factor and
            // conver back to original type
            else
            {
                auto float_conv =
                    prog.insert_instruction(ins, op::convert{shape::float_type}, quant_conv);
                auto l_factor = prog.add_literal(literal(float_conv->get_shape(), vec_factor));
                if(orig_type == shape::float_type)
                {
                    prog.replace_instruction(ins, op::mul{}, l_factor, float_conv);
                }
                else
                {
                    auto adjusted_conv =
                        prog.insert_instruction(ins, op::mul{}, l_factor, float_conv);
                    prog.replace_instruction(ins, op::convert{orig_type}, adjusted_conv);
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

void quantize_int8(program& prog,
                   const target& t,
                   std::vector<program::parameter_map>& calibration_args,
                   const std::vector<std::string>& ins_names)
{
    // insert capture operator
    auto cap_prog          = prog;
    auto int8_quant_params = capture_arguments(cap_prog, t, ins_names);

    // use the calibration data to compute the quantization scale
    cap_prog.compile(t);

    // use all calibration data to run the program to calculate the
    // quantization scale and shift
    for(auto&& arg : calibration_args)
    {
        program::parameter_map m;
        for(auto&& x : cap_prog.get_parameter_shapes())
        {
            if(arg.count(x.first) > 0)
            {
                assert(x.second == arg[x.first].get_shape());
                m[x.first] = t.copy_to(arg[x.first]);
            }
            else
            {
                m[x.first] = t.allocate(x.second);
            }
        }
        cap_prog.eval(m);
    }

    quantize_int8(prog, ins_names, *int8_quant_params);
}

void quantize_int8(program& prog,
                   const target& t,
                   std::vector<program::parameter_map>& calibration_args)
{
    std::vector<std::string> ins_names = {"dot", "convolution"};
    quantize_int8(prog, t, calibration_args, ins_names);
}

// For the input of each input argument, we need to insert a
// capture operator to compute the scale and shift
std::size_t capture_arguments(program& prog,
                              const std::vector<std::string>& ins_names,
                              const std::function<void(std::size_t, std::vector<argument>)>& func)
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

    return num_quant_params;
}

std::shared_ptr<std::vector<std::pair<float, float>>>
capture_arguments(program& prog, const target& t, const std::vector<std::string>& ins_names)
{
    std::shared_ptr<std::vector<std::pair<float, float>>> int8_quant_params =
        std::make_shared<std::vector<std::pair<float, float>>>();
    std::shared_ptr<std::vector<float>> max_abs_vals = std::make_shared<std::vector<float>>();

    auto calc_quant_params = [int8_quant_params, max_abs_vals, &t](std::size_t ins_index,
                                                                   std::vector<argument> args) {
        std::pair<float, float> param_pair{64.0f, 0.0f};

        // scale and shift is need for only int8 type, and we do not
        // consider shift, so set shift to 0
        std::vector<float> vec_val;
        t.copy_from(args.front()).visit([&](auto output) {
            vec_val.assign(output.begin(), output.end());
        });
        auto max_val                = *std::max_element(vec_val.begin(), vec_val.end());
        auto min_val                = *std::min_element(vec_val.begin(), vec_val.end());
        auto max_abs                = std::max(std::fabs(max_val), std::fabs(min_val));
        max_abs_vals->at(ins_index) = std::max(max_abs_vals->at(ins_index), max_abs);

        param_pair.first                 = 127.0f / max_abs_vals->at(ins_index);
        int8_quant_params->at(ins_index) = param_pair;
    };

    auto num_params = capture_arguments(prog, ins_names, calc_quant_params);

    int8_quant_params->resize(num_params, std::pair<float, float>(64.0f, 0.0f));
    max_abs_vals->resize(num_params, 0.0f);

    return int8_quant_params;
}

std::shared_ptr<std::vector<std::pair<float, float>>> capture_arguments(program& prog,
                                                                        const target& t)
{
    std::vector<std::string> ins_names = {"dot", "convolution"};
    return capture_arguments(prog, t, ins_names);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
