/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/compile_gen.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/mlss/mha/gfx1201_mha_64x64x48_64x48x64.hpp>
#include <cctype>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT


struct mlss_compiler : compiler<mlss_compiler>
{
    std::vector<std::string> names() const { return {"mlss_mha"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {

        const auto& device = ctx.get_current_device();
        std::string target_arch = device.get_device_name();

        // auto query_dim  = inputs[0].ndim();
        auto query_lens = inputs[0].lens();
        auto query_strides = inputs[0].strides();

        // auto key_dim = inputs[1].ndim();
        auto key_lens = inputs[1].lens();
        auto key_strides = inputs[1].strides();

        // auto value_dim = inputs[2].ndim();
        auto value_lens = inputs[2].lens();
        auto value_strides = inputs[2].strides();

        auto output_strides = inputs.back().strides();

        for (char &ch : target_arch) {
            ch = std::toupper(ch);
        }
        target_arch = "MLSS_" + target_arch;

        float scale = v.at("scale").to<float>();

        // std::string_view kernelName = multi_head_attention_void_single_pointer_packed_qkv_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1201.m_kernelName;
        // std::array<std::uint8_t, 68984> binaryData = multi_head_attention_void_single_pointer_packed_qkv_128_64x192x48_64x48x64_forward_with_strides_fp16_gfx1201.m_binary;

        std::string_view kernelName = multi_head_attention_void_single_pointer_packed_qkv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1201.m_kernelName;
        std::array<std::uint8_t, 51976> binaryData = multi_head_attention_void_single_pointer_packed_qkv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1201.m_binary;

        std::string kernel_name = std::string(kernelName);
        size_t bin_size = binaryData.size();

        value::binary value_binary(binaryData.data(), bin_size);

        auto nelements  = inputs.back().elements();
        auto block_size = compute_block_size(ctx, nelements, 256);
        hip_compile_options options;
        options.set_launch_params(
            v, compute_global_for(ctx, nelements * block_size, 128), block_size);
        options.output      = inputs.back();
        options.inputs      = inputs;
        options.kernel_name = kernel_name;
        options.output_arg  = inputs.size() - 1;

        std::map<std::string, value> kernel_args{};
        
        kernel_args.emplace("op", static_cast<int>(mlss_op_type::mha));
        kernel_args.emplace("scale", scale);

        return code_object_op{value_binary,
                          kernel_name,
                          options.global,
                          options.local,
                          options.inputs,
                          options.output,
                          options.output_arg,
                          kernel_args};
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {

        auto v = op.to_value();
        auto inputs = ins->inputs();
        auto scale_literal = inputs[3];

        //const float* scale_f = nullptr;
        float scale_f        = 1.0f;
        if(scale_literal->name() == "@literal")
        {
            const char* scale_data = scale_literal->get_literal().data();
            const half* scale_hp   = reinterpret_cast<const half*>(scale_data);
            scale_f                = static_cast<float>(*scale_hp);            
        }

        //v["scale"] = *scale_f;
        v["scale"] = scale_f;
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
