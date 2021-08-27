#include <migraphx/float_equal.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/quantize_int8.hpp>
#include <migraphx/quantize_util.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/op/capture.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/target.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_INT8_QUANTIZATION_PARAMS)

static void ins_quantize_int8(module& modl,
                              instruction_ref ins,
                              std::vector<instruction_ref>& converted_inputs,
                              const std::vector<std::pair<float, float>>& ins_quant_params)
{
    auto orig_type = ins->get_shape().type();
    auto inputs    = ins->inputs();
    if(ins->name() == "dot")
    {
        auto dot_val = ins->get_operator().to_value();
        assert(contains(dot_val, "alpha"));
        assert(contains(dot_val, "beta"));
        auto dot_alpha  = dot_val.at("alpha").to<float>();
        auto dot_beta   = dot_val.at("beta").to<float>();
        float scale_val = dot_alpha / (ins_quant_params[0].first * ins_quant_params[1].first);
        float beta      = (inputs.size() == 3) ? dot_beta : 0.0f;
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
        auto scale      = modl.add_literal(literal(s_scale, vec));
        auto zero_point = modl.add_literal(int32_t(0));
        zero_point      = modl.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", s.lens()}}), zero_point);
        if(inputs.size() == 3 and (not float_equal(beta, 0.0f)))
        {
            auto l_beta = modl.add_literal(-1.0f * beta / scale_val);
            auto m_beta = modl.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", s.lens()}}), l_beta);
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
        auto conv_val  = ins->get_operator().to_value();
        auto scale_val = 1.0 / (ins_quant_params[0].first * ins_quant_params[1].first);

        auto quant_conv =
            modl.insert_instruction(ins, make_op("quant_convolution", conv_val), converted_inputs);

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

void quantize_int8_pass::apply(program& p) const
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

    if(quant_param_index > quant_params.size())
    {
        MIGRAPHX_THROW("QUANTIZE_INT8: number of scales does not match");
    }
}

// For the input of each input argument, we need to insert a
// capture operator to compute the scale and shift
static std::size_t
capture_arguments(module& m,
                  const std::vector<std::string>& ins_names,
                  const std::function<void(std::size_t, std::vector<argument>)>& func,
                  std::size_t param_index)
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
        auto mod_inputs = ins->module_inputs();
        for(auto*& smod : mod_inputs)
        {
            param_index = capture_arguments(*smod, ins_names, func, param_index);
        }

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
                new_ins        = m.insert_instruction(ins, op::capture{param_index++, func}, input);
                ins_map[input] = new_ins;
            }
            new_args.push_back(new_ins);
        }

        instruction::replace(ins, ins->get_operator(), ins->get_shape(), new_args);
    }

    return param_index;
}

void capture_arguments_pass::apply(program& prog) const
{
    auto* mm = prog.get_main_module();
    capture_arguments(*mm, ins_names, f, 0);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
