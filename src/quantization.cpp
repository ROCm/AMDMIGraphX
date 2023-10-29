/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/float_equal.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/quantization.hpp>
#include <migraphx/quantize_fp16.hpp>
#include <migraphx/quantize_int8.hpp>
#include <migraphx/simplify_reshapes.hpp>
#include <migraphx/simplify_qdq.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/optimize_module.hpp>
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

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_INT8_QUANTIZATION_PARAMS)

// This function is to convert any instructions specified in the input
// from double or float to float16 by inserting a convert operator.
// For the conversion, there could be cases of overflowing or underflowing, but it
// is uncommon. Run optimize_module() before converting to fp16 to const eval and fold in FP32 to
// avoid loss of precision.
void quantize_fp16(program& prog, const std::vector<std::string>& ins_names)
{
    run_passes(prog, {optimize_module{}, quantize_fp16_pass{ins_names}, optimize_module{}});
}

void quantize_int8(program& prog,
                   const target& t,
                   const std::vector<parameter_map>& calibration,
                   const std::vector<std::string>& ins_names)
{
    std::set<std::string> op_names = {"convolution", "dot"};
    std::set<std::string> input_ins_names(ins_names.begin(), ins_names.end());
    if(not std::includes(
           op_names.begin(), op_names.end(), input_ins_names.begin(), input_ins_names.end()))
    {
        MIGRAPHX_THROW("QUANTIZE_INT8: only support DOT and CONVOLUTION operation");
    }

    // Run optimize_module() before converting to int8 to const eval and fold in FP32 to
    // avoid loss of precision.
    run_passes(prog, {optimize_module{}});

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

    // pass to add capture argument op
    std::size_t param_num = 0;
    run_passes(prog, {capture_arguments_pass{ins_names, calc_quant_params, &param_num}});
    int8_quant_params->resize(param_num, std::pair<float, float>(64.0f, 0.0f));
    max_abs_vals->resize(param_num, 0.0f);

    // use the calibration data to compute the quantization scale
    auto capture_prog = prog;
    capture_prog.compile(t);

    // use all calibration data to run the program to calculate the
    // quantization scale and shift
    for(auto&& arg : calibration)
    {
        parameter_map m;
        for(auto&& x : capture_prog.get_parameter_shapes())
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
        capture_prog.eval(m);
    }

    // print the quantization parameters in only the main module
    if(enabled(MIGRAPHX_INT8_QUANTIZATION_PARAMS{}))
    {
        for(std::size_t i = 0; i < int8_quant_params->size(); ++i)
        {
            auto param = int8_quant_params->at(i);
            std::cout << "ins_index = " << i << ", scale = " << param.first
                      << ", shift = " << param.second << std::endl;
        }
        std::cout << std::endl;
    }

    run_passes(prog,
               {quantize_int8_pass{ins_names, *int8_quant_params},
                optimize_module{},
                simplify_qdq{},
                dead_code_elimination{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
