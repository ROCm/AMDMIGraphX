#include "migraphx/instruction_ref.hpp"
#include "migraphx/literal.hpp"
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
#include <new>
#include <utility>
#include <set>
#include <iomanip>
#include <migraphx/serialize.hpp>

#include <migraphx/make_op.hpp>

#include <fstream>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_INT8_QUANTIZATION_PARAMS)

static const std::vector<shape::type_t>& get_quantizable_type()
{
    static std::vector<shape::type_t> quantable_types = {
        shape::float_type, shape::double_type, shape::half_type, shape::int32_type};
    return quantable_types;
}

instruction_ref insert_quant_ins(module& modl,
                                 const instruction_ref& insert_loc,
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

    auto ins_s = ins->get_shape();
    assert(contains(get_quantizable_type(), ins_s.type()));
    instruction_ref quant_ins{};
    if(type == shape::int8_type)
    {
        auto zero_point = modl.add_literal(static_cast<int8_t>(shift));
        auto ins_scale  = modl.add_literal(1.0f / scale);
        auto lens       = ins->get_shape().lens();
        ins_scale       = modl.insert_instruction(
            insert_loc, make_op("multibroadcast", {{"output_lens", lens}}), ins_scale);
        zero_point = modl.insert_instruction(
            insert_loc, make_op("multibroadcast", {{"output_lens", lens}}), zero_point);
        quant_ins = modl.insert_instruction(
            insert_loc, make_op("quantizelinear"), ins, ins_scale, zero_point);
    }
    else
    {
        quant_ins =
            modl.insert_instruction(insert_loc, make_op("convert", {{"target_type", type}}), ins);
    }

    map_ins[ins] = quant_ins;

    return quant_ins;
}

void quantize_fp16(module& m,
                   const std::vector<std::string>& ins_names,
                   std::unordered_map<instruction_ref, instruction_ref>& map_fp16,
                   bool include_param = false)
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "@return")
        {
            if(include_param)
            {
                auto inputs = ins->inputs();
                for(auto in : inputs)
                    if(in->get_shape().type() == shape::half_type)
                    {
                        continue;
                    }
                    else if(in->get_shape().type() == shape::float_type or
                            in->get_shape().type() == shape::double_type)
                    {
                        if(in->name() == "convert" and
                           in->inputs().front()->get_shape().type() == shape::half_type)
                        {
                            instruction::replace_argument(ins, in, in->inputs().front());
                        }
                        else
                        {
                            auto in_outs = in->outputs();
                            auto it      = std::find_if(in_outs.begin(), in_outs.end(), [](auto o) {
                                return (o->name() == "convert" and
                                        o->get_shape().type() == shape::half_type);
                            });
                            assert(it != in_outs.end());
                            instruction::replace_argument(ins, in, *it);
                        }
                    }
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
                if(m.has_instruction(input) and include_param)
                {
                    auto param_name = any_cast<builtin::param>(input->get_operator()).parameter;
                    shape s16{shape::half_type, s.lens(), s.strides()};
                    input_fp16 = m.add_parameter(param_name, s16);
                }
                // parameter is in the parent module
                else
                {
                    auto in_outs = input->outputs();
                    auto it      = std::find_if(in_outs.begin(), in_outs.end(), [](auto o) {
                        return (o->name() == "convert" and
                                o->get_shape().type() == shape::half_type);
                    });
                    assert(it != in_outs.end());
                    input_fp16 = *it;
                }
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
            quantize_fp16(*smod, ins_names, map_fp16, true);
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
void quantize_fp16(program& prog, const std::vector<std::string>& ins_names)
{
    auto* mm = prog.get_main_module();
    std::unordered_map<instruction_ref, instruction_ref> map_fp16;
    quantize_fp16(*mm, ins_names, map_fp16, false);
}

static void ins_quantize_int8(module& modl,
                              instruction_ref ins,
                              std::vector<instruction_ref>& converted_inputs,
                              const std::vector<std::pair<float, float>>& ins_quant_params)
{
    auto orig_type = ins->get_shape().type();
    auto inputs    = ins->inputs();
    if(ins->name() == "dot")
    {
        auto dot_op     = any_cast<op::dot>(ins->get_operator());
        float scale_val = dot_op.alpha / (ins_quant_params[0].first * ins_quant_params[1].first);
        float beta      = (inputs.size() == 3) ? dot_op.beta : 0.0f;
        // We need additional checking about the quant_alpha value. If
        // abs(quant_alpha) > 50 (some tmp value set here), we can convert
        // it to an integer as the new_alpha in the quant_dot
        instruction_ref input_c{};
        if(converted_inputs.size() == 3)
        {
            input_c = converted_inputs.back();
            converted_inputs.pop_back();
        }

        auto quant_dot = modl.insert_instruction(
            ins, make_op("quant_dot", {{"alpha", 1}, {"beta", 0}}), converted_inputs);
        auto s = quant_dot->get_shape();

        // wrap scale
        shape s_scale{shape::float_type, s.lens()};
        std::vector<float> vec(s.elements(), scale_val);
        auto scale  = modl.add_literal(literal(s_scale, vec));
        auto l_beta = modl.add_literal(-1.0f * beta / scale_val);
        auto m_beta = modl.insert_instruction(
            ins, make_op("multibroadcast", {{"output_lens", s.lens()}}), l_beta);
        auto zero_point = modl.add_literal(0.0f);
        zero_point      = modl.insert_instruction(
            ins, make_op("multibroadcast", {{"output_lens", s.lens()}}), zero_point);
        if(inputs.size() == 3)
        {
            if(input_c->get_shape().type() != shape::float_type)
            {
                input_c = modl.insert_instruction(
                    ins, make_op("convert", {{"target_type", shape::float_type}}), input_c);
            }
            zero_point = modl.insert_instruction(ins, make_op("mul"), m_beta, input_c);
            if(zero_point->get_shape().type() != s.type())
            {
                zero_point = modl.insert_instruction(
                    ins, make_op("convert", {{"target_type", s.type()}}), zero_point);
            }
        }
        quant_dot =
            modl.insert_instruction(ins, make_op("dequantizelinear"), quant_dot, scale, zero_point);
        if(quant_dot->get_shape().type() != orig_type)
        {
            quant_dot = modl.insert_instruction(
                ins, make_op("convert", {{"target_type", orig_type}}), quant_dot);
        }
        modl.replace_instruction(ins, quant_dot);
    }
    else if(ins->name() == "convolution")
    {
        // Current MIOpen convolution does not support alpha and beta,
        // so we need a separate multiply to adjust the output
        auto conv_op      = any_cast<op::convolution>(ins->get_operator());
        auto padding      = conv_op.padding;
        auto stride       = conv_op.stride;
        auto dilation     = conv_op.dilation;
        auto padding_mode = conv_op.padding_mode;
        auto group        = conv_op.group;
        auto scale_val    = 1.0 / (ins_quant_params[0].first * ins_quant_params[1].first);

        auto quant_conv = modl.insert_instruction(
            ins,
            op::quant_convolution{padding, stride, dilation, padding_mode, group},
            converted_inputs);

        auto s = quant_conv->get_shape();
        std::vector<float> vec_scale(s.elements(), scale_val);
        shape s_scale{shape::float_type, s.lens()};
        auto scale = modl.add_literal(literal(s_scale, vec_scale));
        quant_conv = modl.insert_instruction(ins, make_op("dequantizelinear"), quant_conv, scale);
        if(quant_conv->get_shape().type() != orig_type)
        {
            quant_conv = modl.insert_instruction(
                ins, make_op("convert", {{"target_type", orig_type}}), quant_conv);
        }
        modl.replace_instruction(ins, quant_conv);
    }
    else
    {
        MIGRAPHX_THROW("QUANTIZE_INT8: does not support operator " + ins->name());
    }
}

// int8 quantization is different from fp16 since int8 can only handle value
// -128 ~ 127. To convert the float or double to int8, we need a scale and
// a shift, then the convert can be done as v_int8 = fp * scale + shift.
// To simplify the changes, we consider shift as 0.0f for now.
void quantize_int8_impl(module& m,
                        const std::vector<std::pair<float, float>>& quant_params,
                        const std::vector<std::string>& ins_names,
                        std::unordered_map<instruction_ref, instruction_ref>& map_quant_ins,
                        std::unordered_map<instruction_ref, std::size_t>& map_ins_index,
                        std::size_t& quant_param_index)
{
    const auto& quantizable_types = get_quantizable_type();
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "@return")
            break;

        // if contains subgraph, needs to handle subgraph
        auto mod_inputs = ins->module_inputs();
        for(auto*& smod : mod_inputs)
        {
            quantize_int8_impl(
                *smod, quant_params, ins_names, map_quant_ins, map_ins_index, quant_param_index);
        }

        if(not contains(ins_names, ins->name()))
        {
            continue;
        }

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
            std::size_t ins_index =
                (map_ins_index.count(input) > 0) ? map_ins_index[input] : quant_param_index++;
            map_ins_index[input] = ins_index;

            auto param = quant_params[map_ins_index[input]];
            ins_quant_params.push_back(param);

            // In general, the target_type is int8, but for the dot
            // operation, if it has 3 inputs, then the last one should
            // be converted to int32_type
            shape::type_t quant_type = shape::int8_type;
            auto s                   = input->get_shape();
            if(contains(quantizable_types, s.type()) and s.type() != quant_type and
               (not(inputs.size() == 3 and input == inputs.back())))
            {
                // if the input is a convert operator, uses its input
                // as its current input
                instruction_ref quant_input{};
                if(input->name() == "convert" and
                   input->inputs().front()->get_shape().type() == quant_type)
                {
                    quant_input = input->inputs().front();
                    // the scale in this case is not used, so tune the scale
                    // to 1.0f for this parameter
                    ins_quant_params.back() = std::pair<float, float>(1.0f, 0.0f);
                }
                else
                {
                    quant_input = insert_quant_ins(
                        m, ins, input, quant_type, map_quant_ins, param.first, param.second);
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

        ins_quantize_int8(m, ins, converted_inputs, ins_quant_params);
    }
}

void quantize_int8_impl(program& p,
                        const std::vector<std::pair<float, float>>& quant_params,
                        const std::vector<std::string>& ins_names)
{
    if(enabled(MIGRAPHX_INT8_QUANTIZATION_PARAMS{}))
    {
        for(std::size_t i = 0; i < quant_params.size(); ++i)
        {
            auto param = quant_params.at(i);
            std::cout << "ins_index = " << i << ", scale = " << param.first
                      << ", shift = " << param.second << std::endl;
        }
        std::cout << std::endl;
    }

    // For now, we only support the int8 quantization of gemm and convolution
    std::set<std::string> op_names = {"convolution", "dot"};
    std::set<std::string> input_ins_names(ins_names.begin(), ins_names.end());
    if(!std::includes(
           op_names.begin(), op_names.end(), input_ins_names.begin(), input_ins_names.end()))
    {
        MIGRAPHX_THROW("QUANTIZE_INT8: only support DOT and CONVOLUTION operation");
    }

    auto* mm                      = p.get_main_module();
    std::size_t quant_param_index = 0;
    std::unordered_map<instruction_ref, std::size_t> map_ins_index;
    std::unordered_map<instruction_ref, instruction_ref> map_quant_ins;
    quantize_int8_impl(
        *mm, quant_params, ins_names, map_quant_ins, map_ins_index, quant_param_index);

    if(quant_param_index != quant_params.size())
    {
        MIGRAPHX_THROW("QUANTIZE_INT8: number of scales does not match");
    }
}

void quantize_int8(program& prog,
                   const target& t,
                   const std::vector<parameter_map>& calibration,
                   const std::vector<std::string>& ins_names)
{
    // insert capture operator
    auto cap_prog          = prog;
    auto int8_quant_params = capture_arguments(cap_prog, t, ins_names);

    // use the calibration data to compute the quantization scale
    cap_prog.compile(t);

    // use all calibration data to run the program to calculate the
    // quantization scale and shift
    for(auto&& arg : calibration)
    {
        parameter_map m;
        for(auto&& x : cap_prog.get_parameter_shapes())
        {
            if(arg.count(x.first) > 0)
            {
                assert(x.second == arg.at(x.first).get_shape());
                m[x.first] = t.copy_to(arg.at(x.first));
            }
            else
            {
                m[x.first] = t.allocate(x.second);
            }
        }
        cap_prog.eval(m);
    }

    quantize_int8_impl(prog, *int8_quant_params, ins_names);
}

// For the input of each input argument, we need to insert a
// capture operator to compute the scale and shift
void capture_arguments(module& m,
                       const std::vector<std::string>& ins_names,
                       const std::function<void(std::size_t, std::vector<argument>)>& func,
                       std::size_t& num_quant_params)
{
    // the int8 quantization only support dot and convolution
    std::set<std::string> op_names = {"dot", "convolution"};
    std::set<std::string> input_ins_names(ins_names.begin(), ins_names.end());
    if(!std::includes(
           op_names.begin(), op_names.end(), input_ins_names.begin(), input_ins_names.end()))
    {
        MIGRAPHX_THROW("CAPTURE_ARGUMENTS: input operator is not supported");
    }

    std::unordered_map<instruction_ref, instruction_ref> ins_map;
    for(auto ins : iterator_for(m))
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
                new_ins = m.insert_instruction(
                    std::next(input), op::capture{num_quant_params++, func}, input);
                ins_map[input] = new_ins;
            }
            new_args.push_back(new_ins);
        }

        auto mod_inputs = ins->module_inputs();
        for(auto*& smod : mod_inputs)
        {
            capture_arguments(*smod, ins_names, func, num_quant_params);
        }

        instruction::replace(ins, ins->get_operator(), ins->get_shape(), new_args);
    }
}

std::size_t capture_arguments(program& prog,
                              const std::vector<std::string>& ins_names,
                              const std::function<void(std::size_t, std::vector<argument>)>& func)
{
    auto* mm                     = prog.get_main_module();
    std::size_t num_quant_params = 0;
    capture_arguments(*mm, ins_names, func, num_quant_params);

    return num_quant_params;
}

std::shared_ptr<std::vector<std::pair<float, float>>>
capture_arguments_impl(program& prog, const target& t, const std::vector<std::string>& ins_names)
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
        argument arg = t.copy_from(args.front());
        arg.visit([&](auto output) { vec_val.assign(output.begin(), output.end()); });
        auto max_val                = *std::max_element(vec_val.begin(), vec_val.end());
        auto min_val                = *std::min_element(vec_val.begin(), vec_val.end());
        auto max_abs                = std::max(std::fabs(max_val), std::fabs(min_val));
        max_abs_vals->at(ins_index) = std::max(max_abs_vals->at(ins_index), max_abs);

        // if all values are 0, no need to do scaling
        if(max_abs_vals->at(ins_index) == 0.0f)
        {
            param_pair.first = 1.0f;
        }
        else
        {
            param_pair.first = 127.0f / max_abs_vals->at(ins_index);
        }
        int8_quant_params->at(ins_index) = param_pair;
    };

    auto num_params = capture_arguments(prog, ins_names, calc_quant_params);

    int8_quant_params->resize(num_params, std::pair<float, float>(64.0f, 0.0f));
    max_abs_vals->resize(num_params, 0.0f);

    return int8_quant_params;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
