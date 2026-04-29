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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using namespace migraphx::gpu::gen; // NOLINT

struct mlss_compiler : compiler<mlss_compiler>
{
    std::vector<std::string> names() const { return {"mlss_mha"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        auto query_lens     = inputs[0].lens();
        int batch_size      = query_lens[0];
        int head_num        = query_lens[1];
        int sequence_length = query_lens[2];

        if (starts_with(ctx.get_current_device().get_gfx_name(), "gfx1201"))
        {
            const auto& shader = multi_head_attention_void_single_pointer_packed_qkv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1201;
            std::string kernel_name = std::string(shader.m_kernelName);
            value::binary value_binary(shader.m_binary.data(), shader.m_binary.size());

            std::map<std::string, value> kernel_args{};
            kernel_args.emplace("op", static_cast<int>(mlss_op_type::mha));
            kernel_args.emplace("scale", v.at("scale").to<float>());

            constexpr int grids_per_head = 2; // kernel launches 2 workgroups per (batch, head, seq)
            const int grid = batch_size * head_num * sequence_length * grids_per_head;
            constexpr int mha_block_size = 128;

            hip_compile_options options;
            options.set_launch_params(v, grid, mha_block_size);
            options.output      = inputs.back();
            options.inputs      = inputs;
            options.kernel_name = kernel_name;
            options.output_arg  = inputs.size() - 1;

            return code_object_op{value_binary,
                                kernel_name,
                                options.global,
                                options.local,
                                options.inputs,
                                options.output,
                                options.output_arg,
                                kernel_args};
        }
        else
        {
            MIGRAPHX_THROW("mlss_compiler: unsupported device: " + ctx.get_current_device().get_gfx_name());
        }
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        auto v             = op.to_value();
        auto scale_literal = ins->inputs()[3];

        if(scale_literal->name() != "@literal")
            MIGRAPHX_THROW("mlss_compiler: expected a literal for the scale input, got: " +
                           scale_literal->name());

        const auto* scale_hp = reinterpret_cast<const half*>(scale_literal->get_literal().data());
        float scale          = static_cast<float>(*scale_hp);

        v["scale"] = scale;
        return compile_op(ctx, to_shapes(ins->inputs()), v);
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
