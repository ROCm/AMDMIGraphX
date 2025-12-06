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
#include <migraphx/gather_optimizer.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE
static const char* const gather_kernel = R"__migraphx__(
#include <migraphx/kernels/gather.hpp>
#include <migraphx/kernels/ops.hpp>
#include <migraphx/kernels/integral_constant.hpp>
#include <migraphx/kernels/generic_constant.hpp>
#include <args.hpp>

namespace migraphx {

extern "C" {

MIGRAPHX_GLOBAL void gather_kernel(void* in_data, void* in_indices, void* output) 
{
    make_tensors()(in_data, in_indices, output)([](auto&&... xs) { 
        ${kernel_call}
    });
}

}

} // namespace migraphx

)__migraphx__";

struct gather_compiler : compiler<gather_compiler>
{
    std::vector<std::string> names() const { return {"gather"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        hip_compile_options options;
        const auto& out_s = inputs.back();
        options.inputs         = inputs;
        options.output         = out_s;
        options.kernel_name    = "gather_kernel";
        options.virtual_inputs = inputs;

        auto axis = v.at("axis").to<int>();
        auto axis_str = std::to_string(axis);
        
        // Check if data input is constant (from value hint or default to false)
        bool data_is_constant = v.get("data_is_constant", false);
        
        // Analyze and select the best gather kernel
        auto kernel_func = select_gather_kernel(inputs, axis, data_is_constant);
        
        // Generate the appropriate kernel call based on selected optimization
        std::string kernel_call = kernel_func + "<" + axis_str + ">(xs...);";
        
        // Adjust launch parameters based on kernel type
        if(kernel_func == "gather_opt")
        {
            // Optimized kernel processes 4 elements per thread
            constexpr std::size_t unroll_factor = 4;
            auto global_size = (out_s.elements() + unroll_factor - 1) / unroll_factor;
            options.set_launch_params(v, compute_global_for(ctx, global_size));
        }
        else if(kernel_func == "gather_const_data_opt")
        {
            // Constant data optimized kernel processes 2 elements per thread
            constexpr std::size_t unroll_factor = 2;
            auto global_size = (out_s.elements() + unroll_factor - 1) / unroll_factor;
            options.set_launch_params(v, compute_global_for(ctx, global_size));
        }
        else if(kernel_func == "gather_vectorized")
        {
            // Vectorized kernel processes VecSize elements per iteration
            constexpr std::size_t vec_size = 4;
            auto global_size = (out_s.elements() + vec_size - 1) / vec_size;
            options.set_launch_params(v, compute_global_for(ctx, global_size));
        }
        else
        {
            // Basic, const_data kernels: one thread per element
            options.set_launch_params(v, compute_global_for(ctx, out_s.elements()));
        }

        auto src = interpolate_string(gather_kernel, 
                                     {{"axis", axis_str}, 
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
