/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <migraphx/gpu/compile_hip.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const fused_gather_concat_kernel = R"__migraphx__(
#include <migraphx/kernels/gather_concat.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void fused_gather_concat_kernel(${params}) 
{
    make_tensors()(${args})([](${tensor_args}) { 
        ${kernel_call}
    });
}

}

} // namespace migraphx

)__migraphx__";

struct fused_gather_concat_compiler : compiler<fused_gather_concat_compiler>
{
    std::vector<std::string> names() const { return {"gpu::fused_gather_concat"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        const auto& out_s = inputs.back();
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = "fused_gather_concat_kernel";
        options.virtual_inputs = inputs;

        auto gather_axis = v.at("gather_axis").to<int>();
        auto concat_axis = v.at("concat_axis").to<int>();
        auto num_gathers = v.at("num_gathers").to<std::size_t>();
        
        auto gather_axis_str = std::to_string(gather_axis);
        auto concat_axis_str = std::to_string(concat_axis);
        
        // Build parameter list and argument list
        std::vector<std::string> params;
        std::vector<std::string> args;
        std::vector<std::string> tensor_args;
        
        // Add inputs (data, indices pairs)
        for(std::size_t i = 0; i < num_gathers; ++i)
        {
            params.push_back("void* data" + std::to_string(i));
            params.push_back("void* indices" + std::to_string(i));
            
            args.push_back("data" + std::to_string(i));
            args.push_back("indices" + std::to_string(i));
            
            tensor_args.push_back("auto data" + std::to_string(i));
            tensor_args.push_back("auto indices" + std::to_string(i));
        }
        
        // Add output
        params.push_back("void* output");
        args.push_back("output");
        tensor_args.push_back("auto output");
        
        // Build kernel call based on number of gathers
        std::string kernel_call;
        if(num_gathers == 2)
        {
            kernel_call = "gather_concat_2<" + gather_axis_str + ", " + concat_axis_str + ">("
                         "data0, indices0, data1, indices1, output);";
        }
        else if(num_gathers == 3)
        {
            kernel_call = "gather_concat_3<" + gather_axis_str + ", " + concat_axis_str + ">("
                         "data0, indices0, data1, indices1, data2, indices2, output);";
        }
        else
        {
            // For N > 3, use generic version (less optimized but flexible)
            std::vector<std::string> input_list;
            for(std::size_t i = 0; i < num_gathers; ++i)
            {
                input_list.push_back("data" + std::to_string(i));
                input_list.push_back("indices" + std::to_string(i));
            }
            kernel_call = "gather_concat_n<" + gather_axis_str + ", " + concat_axis_str + ">(output";
            for(const auto& inp : input_list)
            {
                kernel_call += ", " + inp;
            }
            kernel_call += ");";
        }
        
        // Set launch parameters
        options.set_launch_params(v, compute_global_for(ctx, out_s.elements()));
        
        // Generate kernel source
        auto src = interpolate_string(fused_gather_concat_kernel,
                                     {{"params", join_strings(params, ", ")},
                                      {"args", join_strings(args, ", ")},
                                      {"tensor_args", join_strings(tensor_args, ", ")},
                                      {"kernel_call", kernel_call}});
        
        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

