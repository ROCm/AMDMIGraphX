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
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/env.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/gpu/fuse_mlss.hpp>
#include <migraphx/gpu/mlss_mha_op.hpp>
#include <migraphx/gpu/mlss/mha/gfx1201_mha_64x64x48_64x48x64.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

/*
 * Comma-separated list of MLSS ops to enable, e.g. MIGRAPHX_MLSS_USE_SPECIFIC_OPS=mha
 * If unset, no MLSS ops are fused. Recognized values: "mha".
 */
MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_MLSS_USE_SPECIFIC_OPS);

bool mlss_enabled()
{
    return not string_value_of(MIGRAPHX_MLSS_USE_SPECIFIC_OPS{}, "").empty();
}

static bool mlss_op_enabled(std::string_view op_name)
{
    const auto ops = split_string(string_value_of(MIGRAPHX_MLSS_USE_SPECIFIC_OPS{}, ""), ',');
    return std::any_of(ops.begin(), ops.end(), [&](const auto& opt) { return opt == op_name; });
}

void fuse_mlss::apply(module& m) const
{
    const auto& gfx_name = ctx->get_current_device().get_gfx_name();
    if(not starts_with(gfx_name, "gfx1201"))
        return;

    // Supported [batch, heads, seq, head_dim] shapes for the pre-compiled kernels.
    // Add new entries here to enable additional configurations.
    static const std::vector<std::vector<std::size_t>> supported_shapes = {
        {1, 8, 4096, 40},
    };

    if(not mlss_op_enabled("mha"))
        return;

    const auto& shader = multi_head_attention_void_single_pointer_packed_qkv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1201;

    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "group")
            continue;

        auto op_val = ins->get_operator().to_value();
        if(not op_val.contains("tag") or op_val["tag"].to<std::string>() != "attention")
            continue;

        auto& mod_args = ins->module_inputs();
        if(mod_args.empty())
            continue;

        module_ref attn_mod = mod_args[0];

        // Find the half-precision scale literal inside the submodule
        instruction_ref scale_literal_ins = attn_mod->end();
        for(auto sub_ins : iterator_for(*attn_mod))
        {
            if(sub_ins->name() == "@literal")
            {
                scale_literal_ins = sub_ins;
                break;
            }
        }

        if(scale_literal_ins == attn_mod->end())
            continue;

        auto inputs = ins->inputs();
        if(inputs.size() != 3)
            continue;

        auto query_lens = inputs[0]->get_shape().lens();
        bool shape_supported = std::any_of(
            supported_shapes.begin(), supported_shapes.end(), [&](const auto& s) {
                return query_lens == s;
            });

        if(not shape_supported)
            continue;

        const auto* scale_hp =
            reinterpret_cast<const half*>(scale_literal_ins->get_literal().data());
        float scale = static_cast<float>(*scale_hp);

        constexpr int grids_per_head = 2;
        constexpr int mha_block_size = 128;
        int batch_size      = static_cast<int>(query_lens[0]);
        int head_num        = static_cast<int>(query_lens[1]);
        int sequence_length = static_cast<int>(query_lens[2]);

        mlss_mha_op op;
        op.code_object = value::binary(shader.m_binary.data(), shader.m_binary.size());
        op.symbol_name = std::string(shader.m_kernelName);
        op.global      = static_cast<std::size_t>(batch_size * head_num * sequence_length * grids_per_head);
        op.local       = mha_block_size;
        op.scale       = scale;
        op.output      = ins->get_shape();

        // Hoist the scale literal into the parent module
        auto scale_in_main = m.insert_literal(ins, scale_literal_ins->get_literal());

        // Allocate the output buffer — must be an "allocate" node so adjust_allocation
        // can find and validate it via output_alias()
        auto output_alloc = m.insert_instruction(
            ins, make_op("allocate", {{"shape", to_value(op.output)}}));

        m.replace_instruction(ins, op, {inputs[0], inputs[1], inputs[2], scale_in_main, output_alloc});
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
