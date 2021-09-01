#include "migraphx/operation.hpp"
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
#include <numeric>
#include <set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_INT8_QUANTIZATION_PARAMS)

static std::vector<shape::type_t>& get_quantizable_type()
{
    static std::vector<shape::type_t> quantable_types = {
        shape::float_type, shape::double_type, shape::half_type};
    return quantable_types;
}

// // int8 quantization is different from fp16 since int8 can only handle value
// // -128 ~ 127. To convert the float or double to int8, we need a scale and
// // a shift, then the convert can be done as v_int8 = fp * scale + shift.
// // To simplify the changes, we consider shift as 0.0f for now.
// void quantize_int8_impl(module& m,
//                         const std::vector<std::pair<float, float>>& quant_params,
//                         const std::vector<std::string>& ins_names,
//                         // std::unordered_map<instruction_ref, instruction_ref>& map_quant_ins,
//                         std::unordered_map<instruction_ref, std::size_t>& map_ins_index,
//                         std::size_t& quant_param_index)
// {
//     const auto& quantizable_types = get_quantizable_type();
//     for(auto ins : iterator_for(m))
//     {
//         if(ins->name() == "@return")
//             break;

//         // if contains subgraph, needs to handle subgraph
//         auto mod_inputs = ins->module_inputs();
//         for(auto*& smod : mod_inputs)
//         {
//             quantize_int8_impl(*smod, quant_params, ins_names, map_ins_index, quant_param_index);
//         }

//         if(not contains(ins_names, ins->name()))
//         {
//             continue;
//         }

//         // process all inputs, if input is a fp32 or fp64, convert it
//         // to a int8 type by adding a convert operator and replace
//         // the operator with the corresponding int8 version
//         auto inputs = ins->inputs();
//         std::vector<std::pair<float, float>> ins_quant_params;
//         std::vector<instruction_ref> dq_inputs;
//         for(auto input : inputs)
//         {
//             // calculate the index of each instruction to be quantized
//             std::size_t ins_index =
//                 (map_ins_index.count(input) > 0) ? map_ins_index[input] : quant_param_index++;
//             map_ins_index[input] = ins_index;

//             auto param = quant_params[map_ins_index[input]];
//             ins_quant_params.push_back(param);

//             // In general, the target_type is int8, but for the dot
//             // operation, if it has 3 inputs, then the last one should
//             // be converted to int32_type
//             shape::type_t quant_type = shape::int8_type;
//             auto s                   = input->get_shape();
//             if(contains(quantizable_types, s.type()) and s.type() != quant_type)
//             {
//                 auto zero_point = m.add_literal(static_cast<int8_t>(param.second));
//                 auto scale      = m.add_literal(literal({s.type()}, {1.0f / param.first}));
//                 auto lens       = input->get_shape().lens();
//                 scale           = m.insert_instruction(
//                     ins, make_op("multibroadcast", {{"out_lens", lens}}), scale);
//                 zero_point = m.insert_instruction(
//                     ins, make_op("multibroadcast", {{"out_lens", lens}}), zero_point);
//                 auto q_in =
//                     m.insert_instruction(ins, make_op("quantizelinear"), input, scale, zero_point);
//                 auto dq_in =
//                     m.insert_instruction(ins, make_op("dequantizelinear"), q_in, scale, zero_point);
//                 dq_inputs.push_back(dq_in);
//             }
//         }
//         if(inputs != dq_inputs)
//         {
//             m.replace_instruction(ins, ins->get_operator(), dq_inputs);
//         }
//     }
// }

void quantize_int8_pass::apply(module& m) const // NOLINT
{
    const auto& quantizable_types = get_quantizable_type();
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "capture")
            continue;

        auto op_val = ins->get_operator().to_value();
        assert(op_val.contains("ins_index"));

        auto param_index = op_val.at("ins_index").to<std::size_t>();
        auto param = quant_params[param_index];

        auto input = ins->inputs().front();
        auto s = input->get_shape();
        if(contains(quantizable_types, s.type()) and s.type() != shape::int8_type)
        {
            auto zero_point = m.add_literal(static_cast<int8_t>(param.second));
            auto scale      = m.add_literal(literal({s.type()}, {1.0f / param.first}));
            auto lens       = s.lens();
            scale           = m.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", lens}}), scale);
            zero_point = m.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", lens}}), zero_point);
            auto q_in =
                m.insert_instruction(ins, make_op("quantizelinear"), input, scale, zero_point);
            auto dq_in =
                m.insert_instruction(ins, make_op("dequantizelinear"), q_in, scale, zero_point);
            m.replace_instruction(ins, dq_in);
        }

        // if(not contains(ins_names, ins->name()))
        // {
        //     continue;
        // }

        // auto inputs = ins->inputs();
        // std::vector<instruction_ref> converted_inputs;
        // for(auto input : inputs)
        // {
        //     assert(input->name() == "capture");
        //     // replace the capture with the qdq pair
        //     auto op = any_cast<op::capture>(input->get_operator());
        //     auto param_index = op.ins_index;
        //     auto param = quant_params[param_index];

        //     auto in = input->inputs().front();
        //     auto s = in->get_shape();
        //     if(contains(quantizable_types, s.type()) and s.type() != shape::int8_type)
        //     {
        //         auto zero_point = m.add_literal(static_cast<int8_t>(param.second));
        //         auto scale      = m.add_literal(literal({s.type()}, {1.0f / param.first}));
        //         auto lens       = s.lens();
        //         scale           = m.insert_instruction(
        //             ins, make_op("multibroadcast", {{"out_lens", lens}}), scale);
        //         zero_point = m.insert_instruction(
        //             ins, make_op("multibroadcast", {{"out_lens", lens}}), zero_point);
        //         auto q_in =
        //             m.insert_instruction(ins, make_op("quantizelinear"), in, scale, zero_point);
        //         auto dq_in =
        //             m.insert_instruction(ins, make_op("dequantizelinear"), q_in, scale, zero_point);
        //         converted_inputs.push_back(dq_in);
        //     }
        // }
        // m.replace_instruction(ins, ins->get_operator(), converted_inputs);
    }
}

void capture_arguments_pass::apply(module& m) const // NOLINT
{
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
            auto new_in        = m.insert_instruction(ins, op::capture{(*param_index)++, f}, input);
            new_args.push_back(new_in);
        }
        m.replace_instruction(ins, ins->get_operator(), new_args);
    }

}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
