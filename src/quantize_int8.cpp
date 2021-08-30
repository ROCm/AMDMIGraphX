#include <migraphx/float_equal.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/quantize_int8.hpp>
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

const static std::vector<shape::type_t>& get_quantizable_type()
{
    static std::vector<shape::type_t> quantable_types = {
        shape::float_type, shape::double_type, shape::half_type, shape::int32_type};
    return quantable_types;
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
            if(contains(quantizable_types, s.type()) and s.type() != quant_type)
            {
                auto zero_point = m.add_literal(static_cast<int8_t>(param.second));
                auto scale      = m.add_literal(1.0f / param.first);
                auto lens       = input->get_shape().lens();
                scale           = m.insert_instruction(
                    ins, make_op("multibroadcast", {{"out_lens", lens}}), scale);
                zero_point = m.insert_instruction(
                    ins, make_op("multibroadcast", {{"out_lens", lens}}), zero_point);
                auto q_in =
                    m.insert_instruction(ins, make_op("quantizelinear"), input, scale, zero_point);
                auto dq_in =
                    m.insert_instruction(ins, make_op("dequantizelinear"), q_in, scale, zero_point);
                instruction::replace_argument(ins, input, dq_in);
            }
        }
    }
}

void quantize_int8_pass::apply(program& prog) const
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

    auto* mm                      = prog.get_main_module();
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
