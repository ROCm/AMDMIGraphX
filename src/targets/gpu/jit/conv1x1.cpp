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
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compile_hip_code_object.hpp>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

static const char* const conv1x1_kernel = R"__migraphx__(
#include <migraphx/kernels/index.hpp>

namespace migraphx {

extern "C" {

__global__ 
__attribute__((amdgpu_flat_work_group_size(64, 64)))
__attribute__((amdgpu_waves_per_eu(16)))
void conv1x1_kernel(const float* __restrict__ input,
                                          const float* __restrict__ weight,
                                          ${bias_param}
                                          float* __restrict__ output)
{
    const int c_in   = ${c_in};
    const int inp_h  = ${inp_h};
    const int inp_w  = ${inp_w};
    const int out_h  = ${out_h};
    const int out_w  = ${out_w};
    const int stride = ${stride};

    const int out_hw = out_h * out_w;
    const int inp_hw = inp_h * inp_w;

    const unsigned int out_ch = blockIdx.x;
    const unsigned int out_spatial = threadIdx.x;
    
    if (out_spatial >= out_hw) return;
    
    const int out_y = out_spatial / out_w;
    const int out_x = out_spatial % out_w;
    const int in_y = out_y * stride;
    const int in_x = out_x * stride;
    const int in_spatial = in_y * inp_w + in_x;
    
    const float* in_ptr = input + in_spatial;
    const float* w_ptr = weight + out_ch * c_in;
    
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
    
    int ic = 0;
    for(; ic + 7 < c_in; ic += 8) {
        s0 += in_ptr[(ic+0)*inp_hw] * w_ptr[ic+0];
        s1 += in_ptr[(ic+1)*inp_hw] * w_ptr[ic+1];
        s2 += in_ptr[(ic+2)*inp_hw] * w_ptr[ic+2];
        s3 += in_ptr[(ic+3)*inp_hw] * w_ptr[ic+3];
        s4 += in_ptr[(ic+4)*inp_hw] * w_ptr[ic+4];
        s5 += in_ptr[(ic+5)*inp_hw] * w_ptr[ic+5];
        s6 += in_ptr[(ic+6)*inp_hw] * w_ptr[ic+6];
        s7 += in_ptr[(ic+7)*inp_hw] * w_ptr[ic+7];
    }
    for(; ic < c_in; ++ic) {
        s0 += in_ptr[ic*inp_hw] * w_ptr[ic];
    }
    
    float result = (s0 + s1 + s2 + s3) + (s4 + s5 + s6 + s7);
    ${bias_add}
    output[out_ch * out_hw + out_spatial] = result;
}

}

} // namespace migraphx
)__migraphx__";

struct conv1x1_compiler : compiler<conv1x1_compiler>
{
    std::vector<std::string> names() const { return {"gpu::pre_conv1x1"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        bool has_bias = v.get("has_bias", false);
        
        auto input_shape  = inputs[0];
        auto weight_shape = inputs[1];
        auto output_shape = inputs.back();

        auto stride_vec = v.at("strides").to_vector<std::size_t>();
        auto c_in   = input_shape.lens()[1];
        auto c_out  = weight_shape.lens()[0];
        auto inp_h  = input_shape.lens()[2];
        auto inp_w  = input_shape.lens()[3];
        auto out_h  = output_shape.lens()[2];
        auto out_w  = output_shape.lens()[3];
        auto stride = stride_vec[0];

        hip_compile_options options;
        options.inputs = inputs;
        options.output = output_shape;
        options.kernel_name = "conv1x1_kernel";

        options.set_launch_params(v, c_out * 64, 64);

        std::string bias_param = has_bias ? "const float* __restrict__ bias," : "";
        std::string bias_add = has_bias ? "result += bias[out_ch];" : "";
        
        auto src = interpolate_string(conv1x1_kernel,
                                      {{"c_in", std::to_string(c_in)},
                                       {"inp_h", std::to_string(inp_h)},
                                       {"inp_w", std::to_string(inp_w)},
                                       {"out_h", std::to_string(out_h)},
                                       {"out_w", std::to_string(out_w)},
                                       {"stride", std::to_string(stride)},
                                       {"bias_param", bias_param},
                                       {"bias_add", bias_add}});

        return compile_hip_code_object(ctx, src, options);
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        try {
            auto shapes = to_shapes(ins->inputs());
            auto v = op.to_value();
            auto result = compile_op(ctx, shapes, v);
            return result;
        } catch (const std::exception& e) {
            std::cerr << "[conv1x1] EXCEPTION in compile(): " << e.what() << std::endl;
            throw;
        }
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
