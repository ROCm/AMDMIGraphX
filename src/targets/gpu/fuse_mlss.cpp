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
#include <migraphx/pass_manager.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/env.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/gpu/fuse_mlss.hpp>
#include <migraphx/gpu/mlss_mha_op.hpp>
#include <migraphx/gpu/mlss/mha/gfx1201_mha_64x64x48_64x48x64.hpp>
#ifdef MIGRAPHX_USE_AMDMLSS
#include <amdmlss/amdmlss_api.h>
#include <iostream>
#endif

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

struct find_mlss_attention
{
    context* ctx = nullptr;

    auto matcher() const
    {
        return match::name("group")(match::has_op_value("tag", std::string{"attention"}));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;

        auto& mod_args = ins->module_inputs();
        if(mod_args.empty())
            return;

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
            return;

        auto inputs = ins->inputs();
        if(inputs.size() != 3)
            return;

        // Supported [batch, heads, seq, head_dim] shapes for the pre-compiled kernels.
        const std::vector<std::vector<std::size_t>> supported_shapes = {
            {1, 8, 4096, 40},
        };

        auto query_lens = inputs[0]->get_shape().lens();
        bool shape_supported = std::any_of(
            supported_shapes.begin(), supported_shapes.end(), [&](const auto& s) {
                return query_lens == s;
            });

        if(not shape_supported)
            return;

#ifdef MIGRAPHX_USE_AMDMLSS
        {
            const std::string gfx_name = ctx->get_current_device().get_gfx_name();
            MLSScontext mlss_ctx       = 0;
            MLSSstring op_name         = const_cast<MLSSstring>(MLSS_MHA);
            if(mlssCreateContext(&mlss_ctx, const_cast<MLSSstring>(gfx_name.c_str()), op_name) ==
               MLSS_SUCCESS)
            {
                std::uint32_t batch   = static_cast<std::uint32_t>(query_lens[0]);
                std::uint32_t heads   = static_cast<std::uint32_t>(query_lens[1]);
                std::uint32_t q_seq   = static_cast<std::uint32_t>(query_lens[2]);
                std::uint32_t kv_seq  = q_seq;
                std::uint32_t h_dim   = static_cast<std::uint32_t>(query_lens[3]);
                std::uint32_t kv_dim  = 0;
                std::uint32_t packing = MLSS_ATTR_CONFIG_MHA_PACKING_PACKED_QKV;
                float scale_val =
                    1.0f / std::sqrt(static_cast<float>(h_dim)); // placeholder; real scale set below
                MLSSenum dtype = MLSS_FLOAT16;

                mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_MHA_BATCH, &batch);
                mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_MHA_QSEQ, &q_seq);
                mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_MHA_KVSEQ, &kv_seq);
                mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_MHA_KDIM, &kv_dim);
                mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_MHA_VDIM, &kv_dim);
                mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_MHA_SIZEHEADS, &h_dim);
                mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_MHA_PACKING, &packing);
                mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_MHA_HEADCOUNT, &heads);
                mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_MHA_SCALE, &scale_val);
                mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_MHA_DATATYPE, &dtype);

                MLSSstatus* p_statuses = nullptr;
                MLSSsize n_statuses    = 0;
                mlssGetCaps(mlss_ctx, &p_statuses, &n_statuses);

                MLSSbinary* binaries     = nullptr;
                MLSSsize num_binaries    = 0;
                MLSSstatus bin_status    = mlssGetBinaries(mlss_ctx, &binaries, &num_binaries);
                if(bin_status == MLSS_SUCCESS)
                {
                    std::cout << "[fuse_mlss] mlssGetBinaries returned " << num_binaries
                              << " binary variant(s) for " << gfx_name << "\n";
                    for(MLSSsize i = 0; i < num_binaries; ++i)
                    {
                        MLSSvoid* raw_args  = nullptr;
                        MLSSsize arg_count  = 0;
                        MLSSenum arg_type   = 0;
                        mlssVectorRetrieveData(binaries[i].m_argList, &raw_args, &arg_count, &arg_type);

                        std::uint32_t max_indir = 0;
                        if(raw_args != nullptr)
                        {
                            const auto* args = static_cast<const MLSSarg*>(raw_args);
                            for(MLSSsize j = 0; j < arg_count; ++j)
                            {
                                if(args[j].m_isPointer &&
                                   args[j].m_indirectionLevel > max_indir)
                                    max_indir = args[j].m_indirectionLevel;
                            }
                        }
                        std::cout << "  [" << i << "] reloc=" << binaries[i].m_isRelocatable
                                  << "  args=" << arg_count << "  ptr_indir=" << max_indir
                                  << "  size=" << binaries[i].m_binarySize
                                  << "  kernel=" << (binaries[i].m_pKernelName != nullptr
                                                         ? binaries[i].m_pKernelName
                                                         : "<null>")
                                  << "\n";
                    }
                }
                else
                {
                    std::cout << "[fuse_mlss] mlssGetBinaries failed with status " << bin_status
                              << "\n";
                }
            }
        }
#endif

        const auto& shader = multi_head_attention_void_single_pointer_packed_qkv_128_64x64x48_64x48x64_forward_with_strides_fp16_gfx1201;

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

        auto& m = mpm.get_module();

        // Allocate the output buffer — must be an "allocate" node so adjust_allocation
        // can find and validate it via output_alias()
        auto output_alloc = m.insert_instruction(
            ins, make_op("allocate", {{"shape", to_value(ins->get_shape())}}));

        m.replace_instruction(ins, op, {inputs[0], inputs[1], inputs[2], output_alloc});
    }
};

void fuse_mlss::apply(module_pass_manager& mpm) const
{
    const auto& gfx_name = ctx->get_current_device().get_gfx_name();
    if(not starts_with(gfx_name, "gfx1201"))
        return;

    if(not mlss_op_enabled("mha"))
        return;

    match::find_matches(mpm, find_mlss_attention{ctx});
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
