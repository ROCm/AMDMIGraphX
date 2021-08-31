#include <migraphx/float_equal.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/quantize_fp16.hpp>
#include <migraphx/quantize_int8.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/simplify_qdq.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/dead_code_elimination.hpp>
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

// This function is to convert any instructions specified in the input
// from double or float to float16 by inserting a convert operator.
// For the conversion, there could be cases of overflowing, but it
// is very rare in the area of deeping learning, so we just do a
// truncate of the input to get the fp16.
void quantize_fp16(program& prog, const std::vector<std::string>& ins_names)
{
    run_passes(prog,
               {quantize_fp16_pass{ins_names},
                eliminate_common_subexpression{},
                dead_code_elimination{},
                simplify_reshapes{},
                dead_code_elimination{},
                simplify_qdq{},
                dead_code_elimination{}});
}

static std::size_t capture_argument_num(module& m, const std::vector<std::string>& ins_names)
{
    std::size_t num_params = 0;
    for(auto ins : iterator_for(m))
    {
        auto mod_inputs = ins->module_inputs();
        for(auto*& smod : mod_inputs)
        {
            num_params += capture_argument_num(*smod, ins_names);
        }

        if(not contains(ins_names, ins->name()))
        {
            continue;
        }

        num_params += ins->inputs().size();
    }

    return num_params;
}

static std::size_t capture_argument_num(program& prog, const std::vector<std::string>& ins_names)
{
    auto* mm = prog.get_main_module();
    return capture_argument_num(*mm, ins_names);
}

static const std::vector<std::pair<float, float>>
calc_quantize_params(program prog,
                     const target& t,
                     const std::vector<parameter_map>& calibration,
                     const std::vector<std::string>& ins_names)
{
    auto num_quant_params = capture_argument_num(prog, ins_names);

    std::shared_ptr<std::vector<std::pair<float, float>>> int8_quant_params =
        std::make_shared<std::vector<std::pair<float, float>>>(
            num_quant_params, std::pair<float, float>(64.0f, 0.0f));
    std::shared_ptr<std::vector<float>> max_abs_vals =
        std::make_shared<std::vector<float>>(num_quant_params, 0.0f);

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

    // add capture argument op
    run_passes(prog, {capture_arguments_pass{ins_names, calc_quant_params}});

    // use the calibration data to compute the quantization scale
    prog.compile(t);

    // use all calibration data to run the program to calculate the
    // quantization scale and shift
    for(auto&& arg : calibration)
    {
        parameter_map m;
        for(auto&& x : prog.get_parameter_shapes())
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
        prog.eval(m);
    }

    return std::move(*int8_quant_params);
}

void quantize_int8(program& prog,
                   const target& t,
                   const std::vector<parameter_map>& calibration,
                   const std::vector<std::string>& ins_names)
{
    // insert capture operator
    const auto& int8_quant_params = calc_quantize_params(prog, t, calibration, ins_names);
    run_passes(prog,
               {quantize_int8_pass{ins_names, int8_quant_params},
                eliminate_common_subexpression{},
                dead_code_elimination{},
                simplify_reshapes{},
                dead_code_elimination{},
                simplify_qdq{},
                dead_code_elimination{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
