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
static const char* const fused_gather_transpose_kernel = R"__migraphx__(
#include <migraphx/kernels/gather_transpose.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <migraphx/kernels/array.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void fused_gather_transpose_kernel(void* in_data, void* in_indices, void* output) 
{
    constexpr auto perm = make_array(${permutation});
    make_tensors()(in_data, in_indices, output)([&](auto data, auto indices, auto out) { 
        gather_transpose<${gather_axis}>(data, indices, out, perm);
    });
}

}

} // namespace migraphx

)__migraphx__";

struct fused_gather_transpose_compiler : compiler<fused_gather_transpose_compiler>
{
    std::vector<std::string> names() const { return {"gpu::fused_gather_transpose"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        const auto& out_s = inputs.back();
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = "fused_gather_transpose_kernel";
        options.virtual_inputs = inputs;

        auto gather_axis = v.at("gather_axis").to<int>();
        auto permutation = v.at("permutation").to_vector<int64_t>();
        
        // Build permutation string
        std::vector<std::string> perm_strs;
        for(auto p : permutation)
        {
            perm_strs.push_back(std::to_string(p));
        }
        
        options.set_launch_params(v, compute_global_for(ctx, out_s.elements()));
        
        auto src = interpolate_string(fused_gather_transpose_kernel,
                                     {{"gather_axis", std::to_string(gather_axis)},
                                      {"permutation", join_strings(perm_strs, ", ")}});
        
        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

// NOLINTNEXTLINE
static const char* const fused_gather_transpose_concat_kernel = R"__migraphx__(
#include <migraphx/kernels/gather_transpose.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <migraphx/kernels/array.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void fused_gather_transpose_concat_kernel(${params}) 
{
    constexpr auto perm = make_array(${permutation});
    make_tensors()(${args})([&](${tensor_args}) { 
        ${kernel_call}
    });
}

}

} // namespace migraphx

)__migraphx__";

struct fused_gather_transpose_concat_compiler : compiler<fused_gather_transpose_concat_compiler>
{
    std::vector<std::string> names() const { return {"gpu::fused_gather_transpose_concat"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        const auto& out_s = inputs.back();
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = "fused_gather_transpose_concat_kernel";
        options.virtual_inputs = inputs;

        auto gather_axis = v.at("gather_axis").to<int>();
        auto concat_axis = v.at("concat_axis").to<int>();
        auto permutation = v.at("permutation").to_vector<int64_t>();
        auto num_gathers = v.at("num_gathers").to<std::size_t>();
        
        // Build permutation string
        std::vector<std::string> perm_strs;
        for(auto p : permutation)
        {
            perm_strs.push_back(std::to_string(p));
        }
        
        // Build parameter list
        std::vector<std::string> params;
        std::vector<std::string> args;
        std::vector<std::string> tensor_args;
        
        for(std::size_t i = 0; i < num_gathers; ++i)
        {
            params.push_back("void* data" + std::to_string(i));
            params.push_back("void* indices" + std::to_string(i));
            args.push_back("data" + std::to_string(i));
            args.push_back("indices" + std::to_string(i));
            tensor_args.push_back("auto data" + std::to_string(i));
            tensor_args.push_back("auto indices" + std::to_string(i));
        }
        
        params.push_back("void* output");
        args.push_back("output");
        tensor_args.push_back("auto output");
        
        // Build kernel call
        std::string kernel_call;
        if(num_gathers == 2)
        {
            kernel_call = "gather_transpose_concat_2<" + 
                         std::to_string(gather_axis) + ", decltype(perm), " + 
                         std::to_string(concat_axis) + ">(data0, indices0, data1, indices1, output, perm);";
        }
        else if(num_gathers == 3)
        {
            kernel_call = "gather_transpose_concat_3<" + 
                         std::to_string(gather_axis) + ", decltype(perm), " + 
                         std::to_string(concat_axis) + ">(data0, indices0, data1, indices1, data2, indices2, output, perm);";
        }
        else
        {
            // Generic version (less optimal but flexible)
            kernel_call = "// Generic version not yet implemented";
        }
        
        options.set_launch_params(v, compute_global_for(ctx, out_s.elements()));
        
        auto src = interpolate_string(fused_gather_transpose_concat_kernel,
                                     {{"params", join_strings(params, ", ")},
                                      {"args", join_strings(args, ", ")},
                                      {"tensor_args", join_strings(tensor_args, ", ")},
                                      {"permutation", join_strings(perm_strs, ", ")},
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

