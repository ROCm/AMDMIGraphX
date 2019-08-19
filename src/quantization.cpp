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
                                 std::unordered_map<instruction_ref, instruction_ref>& map_ins)
{
    if(map_ins.count(ins) > 0)
    {
        return map_ins[ins];
    }

    if(ins->name() == "undefined")
    {
        return ins;
    }

    assert(ins->get_shape().type() == shape::float_type ||
           ins->get_shape().type() == shape::double_type ||
           ins->get_shape().type() == shape::int32_type);
    instruction_ref quant_ins{};
    quant_ins    = prog.insert_instruction(std::next(ins), op::convert{type}, ins);
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

// For the input of each input argument, we need to insert a
// capture operator to compute the scale and shift
std::size_t capture_arguments(program& prog,
                              const std::vector<std::string>& ins_names,
                              const std::function<void(std::size_t, std::vector<argument>)>& func)
{

    size_t num_quant_params = 0;
    // the int8 quantization only support dot and convolution
    std::vector<std::string> op_names = {"dot", "convolution"};
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
capture_arguments(program& prog, const std::vector<std::string>& ins_names)
{
    std::shared_ptr<std::vector<std::pair<float, float>>> int8_quant_params =
        std::make_shared<std::vector<std::pair<float, float>>>();
    std::shared_ptr<std::vector<float>> max_abs_vals = std::make_shared<std::vector<float>>();

    auto calc_quant_params = [int8_quant_params, max_abs_vals](
                                 std::size_t ins_index, std::vector<migraphx::argument> args) {
        std::pair<float, float> param_pair{64.0f, 0.0f};

        // scale and shift is need for only int8 type, and we do not
        // consider shift, so set shift to 0
        std::vector<float> vec_val;
        args.front().visit([&](auto output) { vec_val.assign(output.begin(), output.end()); });
        auto max_val                = *std::max_element(vec_val.begin(), vec_val.end());
        auto min_val                = *std::min_element(vec_val.begin(), vec_val.end());
        auto max_abs                = std::max(std::fabs(max_val), std::fabs(min_val));
        max_abs_vals->at(ins_index) = std::max(max_abs_vals->at(ins_index), max_abs);

        param_pair.first                 = 127.0f / max_abs_vals->at(ins_index);
        int8_quant_params->at(ins_index) = param_pair;
    };

    auto num_params = capture_arguments(prog, ins_names, calc_quant_params);

    int8_quant_params->resize(num_params, std::make_pair<float, float>(64.0f, 0.0f));
    max_abs_vals->resize(num_params, 0.0f);

    return int8_quant_params;
}

std::shared_ptr<std::vector<std::pair<float, float>>> capture_arguments(program& prog)
{
    std::vector<std::string> ins_names = {"dot", "convolution"};
    return capture_arguments(prog, ins_names);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
