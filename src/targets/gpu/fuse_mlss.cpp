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
#include <migraphx/gpu/mlss_conv_op.hpp>
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

// ---------------------------------------------------------------------------
// Matcher for convolution instructions matching ResNet-50 fp32 shapes.
// The matcher checks the op name and then validates shapes/attributes in apply().
// ---------------------------------------------------------------------------
struct find_mlss_conv
{
    context* ctx = nullptr;

    auto matcher() const
    {
        return match::name("convolution")(
            match::arg(0)(match::any()),
            match::arg(1)(match::name("@literal")));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;

        // Retrieve the two instruction inputs: activation and weight literal.
        auto inputs = ins->inputs();
        if(inputs.size() < 2)
            return;

        auto act_ins = inputs[0];
        auto wt_ins  = inputs[1];

        // ---- shape checks ----
        const auto act_lens = act_ins->get_shape().lens();
        const auto wt_lens  = wt_ins->get_shape().lens();
        const auto out_lens = ins->get_shape().lens();

        const auto dtype = act_ins->get_shape().type();
        if(dtype != shape::float_type and dtype != shape::half_type)
            return;
        if(wt_ins->get_shape().type() != dtype)
            return;
        if(ins->get_shape().type() != dtype)
            return;

        // ---- convolution attribute checks ----
        const auto& op_val = ins->get_operator().to_value();
        auto get_vec = [&](const std::string& key) -> std::vector<std::size_t> {
            return op_val.get(key, std::vector<std::size_t>{});
        };

        if(op_val.get("group", std::size_t{1}) != 1)
            return;
        if(get_vec("dilation") != std::vector<std::size_t>{1, 1})
            return;

        // Each entry: {act_lens, wt_lens, out_lens, padding, stride}
        struct conv_shape_entry
        {
            std::vector<std::size_t> act;
            std::vector<std::size_t> wt;
            std::vector<std::size_t> out;
            std::vector<std::size_t> padding;
            std::vector<std::size_t> stride;
        };

        // fp32 supported shapes (ResNet-50 + VGG-19, stride-1 kernels)
        static const std::vector<conv_shape_entry> fp32_shapes = {
            // ResNet-50 1x1 stride-1 (bottleneck expand/compress)
            // {{1, 64, 56, 56},    {64, 64, 1, 1},    {1, 64, 56, 56},    {0, 0, 0, 0}, {1, 1}},
            // {{1, 64, 56, 56},    {256, 64, 1, 1},   {1, 256, 56, 56},   {0, 0, 0, 0}, {1, 1}},
            // {{1, 256, 56, 56},   {64, 256, 1, 1},   {1, 64, 56, 56},    {0, 0, 0, 0}, {1, 1}},
            // {{1, 128, 28, 28},   {512, 128, 1, 1},  {1, 512, 28, 28},   {0, 0, 0, 0}, {1, 1}},
            // {{1, 512, 28, 28},   {128, 512, 1, 1},  {1, 128, 28, 28},   {0, 0, 0, 0}, {1, 1}},
            // {{1, 256, 14, 14},   {1024, 256, 1, 1}, {1, 1024, 14, 14},  {0, 0, 0, 0}, {1, 1}},
            // {{1, 1024, 14, 14},  {256, 1024, 1, 1}, {1, 256, 14, 14},   {0, 0, 0, 0}, {1, 1}},
            // {{1, 512, 7, 7},     {2048, 512, 1, 1}, {1, 2048, 7, 7},    {0, 0, 0, 0}, {1, 1}},
            // {{1, 2048, 7, 7},    {512, 2048, 1, 1}, {1, 512, 7, 7},     {0, 0, 0, 0}, {1, 1}},

            // ResNet-50 3x3 stride-1
            {{1, 64, 56, 56},    {64, 64, 3, 3},    {1, 64, 56, 56},    {1, 1, 1, 1}, {1, 1}},
            {{1, 128, 28, 28},   {128, 128, 3, 3},  {1, 128, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            {{1, 256, 14, 14},   {256, 256, 3, 3},  {1, 256, 14, 14},   {1, 1, 1, 1}, {1, 1}},

            // vgg19 (224x224)
            {{1, 3, 224, 224},   {64, 3, 3, 3},     {1, 64, 224, 224},  {1, 1, 1, 1}, {1, 1}},
            {{1, 64, 224, 224},  {64, 64, 3, 3},    {1, 64, 224, 224},  {1, 1, 1, 1}, {1, 1}},
            // vgg19 (112x112)
            {{1, 64, 112, 112},  {128, 64, 3, 3},   {1, 128, 112, 112}, {1, 1, 1, 1}, {1, 1}},
            {{1, 128, 112, 112}, {128, 128, 3, 3},  {1, 128, 112, 112}, {1, 1, 1, 1}, {1, 1}},
            // vgg19 (56x56)
            {{1, 128, 56, 56},   {256, 128, 3, 3},  {1, 256, 56, 56},   {1, 1, 1, 1}, {1, 1}},
            {{1, 256, 56, 56},   {256, 256, 3, 3},  {1, 256, 56, 56},   {1, 1, 1, 1}, {1, 1}},
            // vgg19 (28x28)
            {{1, 256, 28, 28},   {512, 256, 3, 3},  {1, 512, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            {{1, 512, 28, 28},   {512, 512, 3, 3},  {1, 512, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            // vgg19 (14x14)
            {{1, 512, 14, 14},   {512, 512, 3, 3},  {1, 512, 14, 14},   {1, 1, 1, 1}, {1, 1}},
            {{1, 512, 7, 7},     {512, 512, 3, 3},  {1, 512, 7, 7},     {1, 1, 1, 1}, {1, 1}},
        };

        // fp16pk supported shapes (NAVI48_fp16pk_f2x3_stride1):
        // R=2, S=3, pad_h=0, pad_w=1 -> out_h = H-1, out_w = W
        static const std::vector<conv_shape_entry> fp16pk_shapes = {
            {{1, 64, 56, 56},    {64, 64, 3, 3},    {1, 64, 56, 56},    {1, 1, 1, 1}, {1, 1}},
            {{1, 128, 28, 28},   {128, 128, 3, 3},  {1, 128, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            {{1, 256, 14, 14},   {256, 256, 3, 3},  {1, 256, 14, 14},   {1, 1, 1, 1}, {1, 1}},

            // vgg19 (224x224)
            {{1, 3, 224, 224},   {64, 3, 3, 3},     {1, 64, 224, 224},  {1, 1, 1, 1}, {1, 1}},
            {{1, 64, 224, 224},  {64, 64, 3, 3},    {1, 64, 224, 224},  {1, 1, 1, 1}, {1, 1}},
            // vgg19 (112x112)
            {{1, 64, 112, 112},  {128, 64, 3, 3},   {1, 128, 112, 112}, {1, 1, 1, 1}, {1, 1}},
            {{1, 128, 112, 112}, {128, 128, 3, 3},  {1, 128, 112, 112}, {1, 1, 1, 1}, {1, 1}},
            // vgg19 (56x56)
            {{1, 128, 56, 56},   {256, 128, 3, 3},  {1, 256, 56, 56},   {1, 1, 1, 1}, {1, 1}},
            {{1, 256, 56, 56},   {256, 256, 3, 3},  {1, 256, 56, 56},   {1, 1, 1, 1}, {1, 1}},
            // vgg19 (28x28)
            {{1, 256, 28, 28},   {512, 256, 3, 3},  {1, 512, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            {{1, 512, 28, 28},   {512, 512, 3, 3},  {1, 512, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            // vgg19 (14x14)
            {{1, 512, 14, 14},   {512, 512, 3, 3},  {1, 512, 14, 14},   {1, 1, 1, 1}, {1, 1}},
            {{1, 512, 7, 7},     {512, 512, 3, 3},  {1, 512, 7, 7},     {1, 1, 1, 1}, {1, 1}},
        };

        // fp32 stride-2 supported shapes (GFX12_fp32_f3x2_ostride2):
        // stride=2; R, S, pad passed as runtime args so any filter size is supported
        static const std::vector<conv_shape_entry> fp32_ostride2_shapes = {
            {{1, 32, 256, 256},  {64, 32, 3, 2},    {1, 64, 128, 128},  {1, 1, 0, 0}, {2, 2}},
            // ResNet-50 stem: 7x7 stride-2
            // {{1, 3, 224, 224},   {64, 3, 7, 7},     {1, 64, 112, 112},  {3, 3, 3, 3}, {2, 2}},
            // ResNet-50 1x1 stride-2 (downsample projections)
            // {{1, 256, 56, 56},   {128, 256, 1, 1},  {1, 128, 28, 28},   {0, 0, 0, 0}, {2, 2}},
            // {{1, 256, 56, 56},   {512, 256, 1, 1},  {1, 512, 28, 28},   {0, 0, 0, 0}, {2, 2}},
            // {{1, 512, 28, 28},   {256, 512, 1, 1},  {1, 256, 14, 14},   {0, 0, 0, 0}, {2, 2}},
            // {{1, 512, 28, 28},   {1024, 512, 1, 1}, {1, 1024, 14, 14},  {0, 0, 0, 0}, {2, 2}},
            // {{1, 1024, 14, 14},  {512, 1024, 1, 1}, {1, 512, 7, 7},     {0, 0, 0, 0}, {2, 2}},
            // {{1, 1024, 14, 14},  {2048, 1024, 1, 1},{1, 2048, 7, 7},    {0, 0, 0, 0}, {2, 2}},
        };

        const auto cur_padding = get_vec("padding");
        const auto cur_stride  = get_vec("stride");

        auto shape_match = [&](const std::vector<conv_shape_entry>& table) {
            return std::any_of(table.begin(), table.end(), [&](const conv_shape_entry& e) {
                return act_lens == e.act and wt_lens == e.wt and out_lens == e.out and
                       cur_padding == e.padding and cur_stride == e.stride;
            });
        };

        // ---- build the mlss_conv_op ----
        mlss_conv_op op;
        if(dtype == shape::float_type)
        {
            // if(shape_match(fp32_ostride2_shapes))
            //     op = mlss_conv_op::make_gfx12_fp32_f3x2_ostride2();
            if(shape_match(fp32_shapes))
                op = mlss_conv_op::make_gfx12_fp32_f2x3_stride1();
            else
                return;
        }
        else // half_type
        {
            if(not shape_match(fp16pk_shapes))
                return;
            op = mlss_conv_op::make_navi48_fp16pk_f2x3_stride1();
        }

        // Set pad_h and pad_w from the convolution padding attribute.
        // MIGraphX padding layout: {pad_h_begin, pad_w_begin, pad_h_end, pad_w_end}
        op.pad_h = static_cast<int32_t>(cur_padding[0]);
        op.pad_w = static_cast<int32_t>(cur_padding[1]);

        auto& m = mpm.get_module();

        const auto out_shape  = ins->get_shape();

        // Output buffer
        auto output_alloc = m.insert_instruction(
            ins, make_op("allocate", {{"shape", to_value(out_shape)}}));

        m.replace_instruction(ins, op, {act_ins, wt_ins, output_alloc});
    }
};

// ---------------------------------------------------------------------------
// Matcher for conv+bias pattern:
//   add(convolution(input, weight_literal), broadcast(bias_literal))
// Fuses the add into mlss_conv_op with has_bias=true, enabling the kernel's
// built-in bias add (F_BIAS flag, bit 7 of flags64).
// The matched instruction is the "add"; its output shape is the same as the
// convolution output, so no shape change is needed.
// ---------------------------------------------------------------------------
struct find_mlss_conv_bias
{
    context* ctx = nullptr;

    auto matcher() const
    {
        // Match: add( convolution(any, @literal), broadcast(@literal) )
        auto conv_with_literal_weight =
            match::name("convolution")(
                match::arg(0)(match::any()),
                match::arg(1)(match::name("@literal")));

        auto broadcast_of_literal =
            match::name("broadcast")(
                match::arg(0)(match::name("@literal")));

        return match::name("add")(
            match::arg(0)(conv_with_literal_weight),
            match::arg(1)(broadcast_of_literal));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto add_ins  = r.result;
        auto conv_ins = add_ins->inputs()[0]; // convolution
        auto bcast_ins = add_ins->inputs()[1]; // broadcast

        // The broadcast's input is the raw bias literal {K}
        auto bias_ins = bcast_ins->inputs()[0];

        // Retrieve convolution inputs
        auto conv_inputs = conv_ins->inputs();
        if(conv_inputs.size() < 2)
            return;
        auto act_ins = conv_inputs[0];
        auto wt_ins  = conv_inputs[1];

        // ---- shape checks (same as find_mlss_conv) ----
        const auto act_lens = act_ins->get_shape().lens();
        const auto wt_lens  = wt_ins->get_shape().lens();
        const auto out_lens = conv_ins->get_shape().lens(); // same as add output

        const auto dtype = act_ins->get_shape().type();
        if(dtype != shape::float_type and dtype != shape::half_type)
            return;
        if(wt_ins->get_shape().type() != dtype)
            return;
        if(conv_ins->get_shape().type() != dtype)
            return;

        // ---- convolution attribute checks ----
        const auto& op_val = conv_ins->get_operator().to_value();
        auto get_vec = [&](const std::string& key) -> std::vector<std::size_t> {
            return op_val.get(key, std::vector<std::size_t>{});
        };

        if(op_val.get("group", std::size_t{1}) != 1)
            return;
        if(get_vec("dilation") != std::vector<std::size_t>{1, 1})
            return;

        const auto cur_padding = get_vec("padding");
        const auto cur_stride  = get_vec("stride");

        // Reuse the same shape tables from find_mlss_conv.
        struct conv_shape_entry
        {
            std::vector<std::size_t> act;
            std::vector<std::size_t> wt;
            std::vector<std::size_t> out;
            std::vector<std::size_t> padding;
            std::vector<std::size_t> stride;
        };

        static const std::vector<conv_shape_entry> fp32_shapes = {
            {{1, 64, 56, 56},    {64, 64, 3, 3},    {1, 64, 56, 56},    {1, 1, 1, 1}, {1, 1}},
            {{1, 128, 28, 28},   {128, 128, 3, 3},  {1, 128, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            {{1, 256, 14, 14},   {256, 256, 3, 3},  {1, 256, 14, 14},   {1, 1, 1, 1}, {1, 1}},
            {{1, 3, 224, 224},   {64, 3, 3, 3},     {1, 64, 224, 224},  {1, 1, 1, 1}, {1, 1}},
            {{1, 64, 224, 224},  {64, 64, 3, 3},    {1, 64, 224, 224},  {1, 1, 1, 1}, {1, 1}},
            {{1, 64, 112, 112},  {128, 64, 3, 3},   {1, 128, 112, 112}, {1, 1, 1, 1}, {1, 1}},
            {{1, 128, 112, 112}, {128, 128, 3, 3},  {1, 128, 112, 112}, {1, 1, 1, 1}, {1, 1}},
            {{1, 128, 56, 56},   {256, 128, 3, 3},  {1, 256, 56, 56},   {1, 1, 1, 1}, {1, 1}},
            {{1, 256, 56, 56},   {256, 256, 3, 3},  {1, 256, 56, 56},   {1, 1, 1, 1}, {1, 1}},
            {{1, 256, 28, 28},   {512, 256, 3, 3},  {1, 512, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            {{1, 512, 28, 28},   {512, 512, 3, 3},  {1, 512, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            {{1, 512, 14, 14},   {512, 512, 3, 3},  {1, 512, 14, 14},   {1, 1, 1, 1}, {1, 1}},
            {{1, 512, 7, 7},     {512, 512, 3, 3},  {1, 512, 7, 7},     {1, 1, 1, 1}, {1, 1}},
        };

        static const std::vector<conv_shape_entry> fp16pk_shapes = {
            {{1, 64, 56, 56},    {64, 64, 3, 3},    {1, 64, 56, 56},    {1, 1, 1, 1}, {1, 1}},
            {{1, 128, 28, 28},   {128, 128, 3, 3},  {1, 128, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            {{1, 256, 14, 14},   {256, 256, 3, 3},  {1, 256, 14, 14},   {1, 1, 1, 1}, {1, 1}},
            {{1, 3, 224, 224},   {64, 3, 3, 3},     {1, 64, 224, 224},  {1, 1, 1, 1}, {1, 1}},
            {{1, 64, 224, 224},  {64, 64, 3, 3},    {1, 64, 224, 224},  {1, 1, 1, 1}, {1, 1}},
            {{1, 64, 112, 112},  {128, 64, 3, 3},   {1, 128, 112, 112}, {1, 1, 1, 1}, {1, 1}},
            {{1, 128, 112, 112}, {128, 128, 3, 3},  {1, 128, 112, 112}, {1, 1, 1, 1}, {1, 1}},
            {{1, 128, 56, 56},   {256, 128, 3, 3},  {1, 256, 56, 56},   {1, 1, 1, 1}, {1, 1}},
            {{1, 256, 56, 56},   {256, 256, 3, 3},  {1, 256, 56, 56},   {1, 1, 1, 1}, {1, 1}},
            {{1, 256, 28, 28},   {512, 256, 3, 3},  {1, 512, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            {{1, 512, 28, 28},   {512, 512, 3, 3},  {1, 512, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            {{1, 512, 14, 14},   {512, 512, 3, 3},  {1, 512, 14, 14},   {1, 1, 1, 1}, {1, 1}},
            {{1, 512, 7, 7},     {512, 512, 3, 3},  {1, 512, 7, 7},     {1, 1, 1, 1}, {1, 1}},
        };

        auto shape_match = [&](const std::vector<conv_shape_entry>& table) {
            return std::any_of(table.begin(), table.end(), [&](const conv_shape_entry& e) {
                return act_lens == e.act and wt_lens == e.wt and out_lens == e.out and
                       cur_padding == e.padding and cur_stride == e.stride;
            });
        };

        // ---- build the mlss_conv_op ----
        mlss_conv_op op;
        if(dtype == shape::float_type)
        {
            if(shape_match(fp32_shapes))
                op = mlss_conv_op::make_gfx12_fp32_f2x3_stride1();
            else
                return;
        }
        else // half_type
        {
            if(not shape_match(fp16pk_shapes))
                return;
            op = mlss_conv_op::make_navi48_fp16pk_f2x3_stride1();
        }

        op.pad_h    = static_cast<int32_t>(cur_padding[0]);
        op.pad_w    = static_cast<int32_t>(cur_padding[1]);
        op.has_bias = true;

        auto& m = mpm.get_module();

        const auto out_shape = add_ins->get_shape();
        auto output_alloc = m.insert_instruction(
            add_ins, make_op("allocate", {{"shape", to_value(out_shape)}}));

        // args: [input, weight, bias, output]
        // replace_instruction rewrites add_ins in-place and detaches bcast_ins
        // and conv_ins from its input list, leaving them with no users.
        m.replace_instruction(add_ins, op, {act_ins, wt_ins, bias_ins, output_alloc});

        // Remove now-dead instructions (outputs().empty() guards against the
        // unlikely case another instruction also consumed them).
        if(bcast_ins->outputs().empty())
            m.remove_instruction(bcast_ins);
        if(conv_ins->outputs().empty())
            m.remove_instruction(conv_ins);
    }
};

// ---------------------------------------------------------------------------
// Matcher for conv+bias+relu pattern:
//   relu(add(convolution(input, weight_literal), broadcast(bias_literal)))
// Same as find_mlss_conv_bias but sets activation_mode=4 (ReLU) and matches
// the relu instruction as the outermost node to replace.
// ---------------------------------------------------------------------------
struct find_mlss_conv_bias_relu
{
    context* ctx = nullptr;

    auto matcher() const
    {
        auto conv_with_literal_weight =
            match::name("convolution")(
                match::arg(0)(match::any()),
                match::arg(1)(match::name("@literal")));

        auto broadcast_of_literal =
            match::name("broadcast")(
                match::arg(0)(match::name("@literal")));

        auto add_conv_bias =
            match::name("add")(
                match::arg(0)(conv_with_literal_weight),
                match::arg(1)(broadcast_of_literal));

        return match::name("relu")(match::arg(0)(add_conv_bias));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto relu_ins  = r.result;
        auto add_ins   = relu_ins->inputs()[0];   // add
        auto conv_ins  = add_ins->inputs()[0];    // convolution
        auto bcast_ins = add_ins->inputs()[1];    // broadcast
        auto bias_ins  = bcast_ins->inputs()[0];  // @literal {K}

        auto conv_inputs = conv_ins->inputs();
        if(conv_inputs.size() < 2)
            return;
        auto act_ins = conv_inputs[0];
        auto wt_ins  = conv_inputs[1];

        // ---- shape / attribute checks (same as find_mlss_conv_bias) ----
        const auto act_lens = act_ins->get_shape().lens();
        const auto wt_lens  = wt_ins->get_shape().lens();
        const auto out_lens = conv_ins->get_shape().lens();

        const auto dtype = act_ins->get_shape().type();
        if(dtype != shape::float_type and dtype != shape::half_type)
            return;
        if(wt_ins->get_shape().type() != dtype)
            return;
        if(conv_ins->get_shape().type() != dtype)
            return;

        const auto& op_val = conv_ins->get_operator().to_value();
        auto get_vec = [&](const std::string& key) -> std::vector<std::size_t> {
            return op_val.get(key, std::vector<std::size_t>{});
        };
        if(op_val.get("group", std::size_t{1}) != 1)
            return;
        if(get_vec("dilation") != std::vector<std::size_t>{1, 1})
            return;

        const auto cur_padding = get_vec("padding");
        const auto cur_stride  = get_vec("stride");

        struct conv_shape_entry
        {
            std::vector<std::size_t> act;
            std::vector<std::size_t> wt;
            std::vector<std::size_t> out;
            std::vector<std::size_t> padding;
            std::vector<std::size_t> stride;
        };

        static const std::vector<conv_shape_entry> fp32_shapes = {
            {{1, 64, 56, 56},    {64, 64, 3, 3},    {1, 64, 56, 56},    {1, 1, 1, 1}, {1, 1}},
            {{1, 128, 28, 28},   {128, 128, 3, 3},  {1, 128, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            {{1, 256, 14, 14},   {256, 256, 3, 3},  {1, 256, 14, 14},   {1, 1, 1, 1}, {1, 1}},
            {{1, 3, 224, 224},   {64, 3, 3, 3},     {1, 64, 224, 224},  {1, 1, 1, 1}, {1, 1}},
            {{1, 64, 224, 224},  {64, 64, 3, 3},    {1, 64, 224, 224},  {1, 1, 1, 1}, {1, 1}},
            {{1, 64, 112, 112},  {128, 64, 3, 3},   {1, 128, 112, 112}, {1, 1, 1, 1}, {1, 1}},
            {{1, 128, 112, 112}, {128, 128, 3, 3},  {1, 128, 112, 112}, {1, 1, 1, 1}, {1, 1}},
            {{1, 128, 56, 56},   {256, 128, 3, 3},  {1, 256, 56, 56},   {1, 1, 1, 1}, {1, 1}},
            {{1, 256, 56, 56},   {256, 256, 3, 3},  {1, 256, 56, 56},   {1, 1, 1, 1}, {1, 1}},
            {{1, 256, 28, 28},   {512, 256, 3, 3},  {1, 512, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            {{1, 512, 28, 28},   {512, 512, 3, 3},  {1, 512, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            {{1, 512, 14, 14},   {512, 512, 3, 3},  {1, 512, 14, 14},   {1, 1, 1, 1}, {1, 1}},
            {{1, 512, 7, 7},     {512, 512, 3, 3},  {1, 512, 7, 7},     {1, 1, 1, 1}, {1, 1}},
        };

        static const std::vector<conv_shape_entry> fp16pk_shapes = {
            {{1, 64, 56, 56},    {64, 64, 3, 3},    {1, 64, 56, 56},    {1, 1, 1, 1}, {1, 1}},
            {{1, 128, 28, 28},   {128, 128, 3, 3},  {1, 128, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            {{1, 256, 14, 14},   {256, 256, 3, 3},  {1, 256, 14, 14},   {1, 1, 1, 1}, {1, 1}},
            {{1, 3, 224, 224},   {64, 3, 3, 3},     {1, 64, 224, 224},  {1, 1, 1, 1}, {1, 1}},
            {{1, 64, 224, 224},  {64, 64, 3, 3},    {1, 64, 224, 224},  {1, 1, 1, 1}, {1, 1}},
            {{1, 64, 112, 112},  {128, 64, 3, 3},   {1, 128, 112, 112}, {1, 1, 1, 1}, {1, 1}},
            {{1, 128, 112, 112}, {128, 128, 3, 3},  {1, 128, 112, 112}, {1, 1, 1, 1}, {1, 1}},
            {{1, 128, 56, 56},   {256, 128, 3, 3},  {1, 256, 56, 56},   {1, 1, 1, 1}, {1, 1}},
            {{1, 256, 56, 56},   {256, 256, 3, 3},  {1, 256, 56, 56},   {1, 1, 1, 1}, {1, 1}},
            {{1, 256, 28, 28},   {512, 256, 3, 3},  {1, 512, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            {{1, 512, 28, 28},   {512, 512, 3, 3},  {1, 512, 28, 28},   {1, 1, 1, 1}, {1, 1}},
            {{1, 512, 14, 14},   {512, 512, 3, 3},  {1, 512, 14, 14},   {1, 1, 1, 1}, {1, 1}},
            {{1, 512, 7, 7},     {512, 512, 3, 3},  {1, 512, 7, 7},     {1, 1, 1, 1}, {1, 1}},
        };

        auto shape_match = [&](const std::vector<conv_shape_entry>& table) {
            return std::any_of(table.begin(), table.end(), [&](const conv_shape_entry& e) {
                return act_lens == e.act and wt_lens == e.wt and out_lens == e.out and
                       cur_padding == e.padding and cur_stride == e.stride;
            });
        };

        mlss_conv_op op;
        if(dtype == shape::float_type)
        {
            if(shape_match(fp32_shapes))
                op = mlss_conv_op::make_gfx12_fp32_f2x3_stride1();
            else
                return;
        }
        else
        {
            if(not shape_match(fp16pk_shapes))
                return;
            op = mlss_conv_op::make_navi48_fp16pk_f2x3_stride1();
        }

        op.pad_h           = static_cast<int32_t>(cur_padding[0]);
        op.pad_w           = static_cast<int32_t>(cur_padding[1]);
        op.has_bias        = true;
        op.activation_mode = 4; // ReLU

        auto& m = mpm.get_module();

        // Output shape matches the relu output (same as conv/add output).
        const auto out_shape  = relu_ins->get_shape();
        auto output_alloc = m.insert_instruction(
            relu_ins, make_op("allocate", {{"shape", to_value(out_shape)}}));

        // Replace the relu instruction; conv, add, broadcast become dead.
        m.replace_instruction(relu_ins, op, {act_ins, wt_ins, bias_ins, output_alloc});

        // Remove now-dead instructions inner-to-outer.
        if(add_ins->outputs().empty())
            m.remove_instruction(add_ins);
        if(bcast_ins->outputs().empty())
            m.remove_instruction(bcast_ins);
        if(conv_ins->outputs().empty())
            m.remove_instruction(conv_ins);
    }
};

void fuse_mlss::apply(module_pass_manager& mpm) const
{
    const auto& gfx_name = ctx->get_current_device().get_gfx_name();
    if(not starts_with(gfx_name, "gfx1201"))
        return;

    if(mlss_op_enabled("mha"))
        match::find_matches(mpm, find_mlss_attention{ctx});

    if(mlss_op_enabled("conv"))
    {
        // Match most-specific patterns first to avoid partial consumption.
        match::find_matches(mpm, find_mlss_conv_bias_relu{ctx});
        match::find_matches(mpm, find_mlss_conv_bias{ctx});
        match::find_matches(mpm, find_mlss_conv{ctx});
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
