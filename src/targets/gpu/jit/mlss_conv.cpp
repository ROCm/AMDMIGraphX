/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/gpu/code_object_op.hpp>
#include <migraphx/gpu/mlss_conv_op.hpp>
#include <cstring>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Store a scalar kernel argument as a value::binary blob preserving the exact byte width.
// binary.size() encodes sizeof(T), so no parallel size map is needed.
template <class T>
static void set_karg(std::map<std::size_t, value>& ka, std::size_t idx, T v)
{
    value::binary b(sizeof(T));
    std::memcpy(b.data(), &v, sizeof(T));
    ka[idx] = value(std::move(b));
}

struct mlss_conv_compiler : compiler<mlss_conv_compiler>
{
    std::vector<std::string> names() const { return {"mlss_conv"}; }

    operation compile_op(context& ctx, const std::vector<shape>& inputs, const value& v) const
    {
        // Extract metadata from the intermediate mlss_conv op
        auto    cur_padding      = v.at("padding").to_vector<std::size_t>();
        bool    has_bias         = v.at("has_bias").to<bool>();
        uint8_t activation_mode  = static_cast<uint8_t>(v.at("activation_mode").to<uint64_t>());
        float   activation_alpha = static_cast<float>(v.at("activation_alpha").to<double>());

        // Input shapes:
        //   No bias:  [input, weight, output_buffer]
        //   Has bias: [input, weight, bias, output_buffer]
        const auto& input_shape  = inputs[0];
        const auto& weight_shape = inputs[1];
        const auto& out_shape    = inputs.back();

        const auto in_lens  = input_shape.lens();   // N, C, H, W
        const auto wt_lens  = weight_shape.lens();  // K, C/g, R, S
        const auto out_lens = out_shape.lens();      // N, K, OH, OW

        // Query AMDMLSS API for kernel binary
        const auto cur_stride   = v.at("stride").to_vector<std::size_t>();
        const auto cur_dilation = v.at("dilation").to_vector<std::size_t>();
        const auto cur_group    = v.at("group").to<std::size_t>();

        auto info = query_mlss_conv_binary(
            ctx, in_lens, wt_lens, out_lens, cur_padding, cur_stride,
            cur_dilation, cur_group,
            has_bias, activation_mode, input_shape.type());
        if(info.empty())
            MIGRAPHX_THROW("mlss_conv_compiler: no AMDMLSS binary for this configuration");

        // -----------------------------------------------------------------------
        // Build kernel_args map — matches the 0xe8-byte kernarg layout from
        // kernel_execution_conv_fp32_f2x3_stride1_cg64_kg128.cpp
        // -----------------------------------------------------------------------
        std::map<std::size_t, value> kernel_args;
        std::map<std::size_t, std::size_t> runtime_arg_indices;

        int32_t N     = static_cast<int32_t>(in_lens[0]);
        int32_t Cg    = static_cast<int32_t>(in_lens[1]);
        int32_t H     = static_cast<int32_t>(in_lens[2]);
        int32_t W     = static_cast<int32_t>(in_lens[3]);
        int32_t Kg    = static_cast<int32_t>(wt_lens[0]);
        int32_t R     = static_cast<int32_t>(wt_lens[2]);
        int32_t S     = static_cast<int32_t>(wt_lens[3]);
        int32_t out_h = static_cast<int32_t>(out_lens[2]);
        int32_t out_w = static_cast<int32_t>(out_lens[3]);
        int32_t G     = 1;
        int32_t ng    = static_cast<int32_t>(info.n_groups);

        // Cap ng to prevent idle workgroups from writing out-of-bounds.
        {
            const int32_t kg_per_workgroup = 128;
            int32_t k_groups    = (Kg + kg_per_workgroup - 1) / kg_per_workgroup;
            int32_t h_tiles     = (out_h + 1) / 2;
            int32_t w_tiles     = (out_w + 1) / 2;
            int32_t total_tiles = N * G * h_tiles * w_tiles * k_groups;
            if(ng > total_tiles)
                ng = total_tiles;
        }

        // flags64 encoding
        uint64_t flags64 = has_bias
            ? ((uint64_t{1} << 7) | (uint64_t{1} << 9) | (uint64_t{1} << 14) | (uint64_t{1} << 15))
            : (uint64_t{1} << 10);

        float alpha = activation_alpha;
        float beta  = 0.0f;

        // Strides (NCHW)
        const auto in_strides  = input_shape.strides();
        const auto wt_strides  = weight_shape.strides();
        const auto out_strides = out_shape.strides();

        int32_t d_N_stride = static_cast<int32_t>(in_strides[0]);
        int32_t d_C_stride = static_cast<int32_t>(in_strides[1]);
        int32_t d_H_stride = static_cast<int32_t>(in_strides[2]);
        int32_t d_G_stride = d_N_stride;

        int32_t f_K_stride = static_cast<int32_t>(wt_strides[0]);
        int32_t f_C_stride = static_cast<int32_t>(wt_strides[1]);
        int32_t f_R_stride = static_cast<int32_t>(wt_strides[2]);
        int32_t f_G_stride = Kg * f_K_stride;

        int32_t o_N_stride = static_cast<int32_t>(out_strides[0]);
        int32_t o_K_stride = static_cast<int32_t>(out_strides[1]);
        int32_t o_H_stride = static_cast<int32_t>(out_strides[2]);
        int32_t o_G_stride = o_N_stride;

        // 0x00-0x14: N, Cg, H, W, Kg, ng
        set_karg(kernel_args, 0, N);
        set_karg(kernel_args, 1, Cg);
        set_karg(kernel_args, 2, H);
        set_karg(kernel_args, 3, W);
        set_karg(kernel_args, 4, Kg);
        set_karg(kernel_args, 5, ng);
        // 0x18: flags64
        set_karg(kernel_args, 6, flags64);
        // 0x20: p_data (runtime arg 0)
        set_karg(kernel_args, 7, uint64_t{0});
        runtime_arg_indices[7] = 0;
        // 0x28: p_filter (runtime arg 1)
        set_karg(kernel_args, 8, uint64_t{0});
        runtime_arg_indices[8] = 1;
        // 0x30: p_output (runtime arg: last)
        set_karg(kernel_args, 9, uint64_t{0});
        runtime_arg_indices[9] = inputs.size() - 1;
        // 0x38: reserved3
        set_karg(kernel_args, 10, uint64_t{0});
        int32_t pad_h = cur_padding.size() > 0 ? static_cast<int32_t>(cur_padding[0]) : 0;
        int32_t pad_w = cur_padding.size() > 1 ? static_cast<int32_t>(cur_padding[1]) : 0;

        // 0x40-0x54: R, S, pad_h, pad_w, out_h, out_w
        set_karg(kernel_args, 11, R);
        set_karg(kernel_args, 12, S);
        set_karg(kernel_args, 13, pad_h);
        set_karg(kernel_args, 14, pad_w);
        set_karg(kernel_args, 15, out_h);
        set_karg(kernel_args, 16, out_w);
        // 0x58: p_bias (runtime arg 2 if has_bias, else 0)
        set_karg(kernel_args, 17, uint64_t{0});
        if(has_bias)
            runtime_arg_indices[17] = 2;
        // 0x60: alpha, beta
        set_karg(kernel_args, 18, alpha);
        set_karg(kernel_args, 19, beta);
        // 0x68-0x87: 8 x zero32 (d/f/o/b offsets, unused)
        for(std::size_t i = 20; i < 28; ++i)
            set_karg(kernel_args, i, uint32_t{0});
        // 0x88-0x94: d_N, d_C, d_H strides + reserved4
        set_karg(kernel_args, 28, d_N_stride);
        set_karg(kernel_args, 29, d_C_stride);
        set_karg(kernel_args, 30, d_H_stride);
        set_karg(kernel_args, 31, uint32_t{0}); // reserved4
        // 0x98-0xa4: f_K, f_C, f_R strides + reserved5
        set_karg(kernel_args, 32, f_K_stride);
        set_karg(kernel_args, 33, f_C_stride);
        set_karg(kernel_args, 34, f_R_stride);
        set_karg(kernel_args, 35, uint32_t{0}); // reserved5
        // 0xa8-0xb4: o_N, o_K, o_H strides + reserved6
        set_karg(kernel_args, 36, o_N_stride);
        set_karg(kernel_args, 37, o_K_stride);
        set_karg(kernel_args, 38, o_H_stride);
        set_karg(kernel_args, 39, uint32_t{0}); // reserved6
        // 0xb8-0xc4: G, d_G, f_G, o_G strides
        set_karg(kernel_args, 40, G);
        set_karg(kernel_args, 41, d_G_stride);
        set_karg(kernel_args, 42, f_G_stride);
        set_karg(kernel_args, 43, o_G_stride);
        // 0xc8-0xe7: activation + sync fields (always present for bias path)
        set_karg(kernel_args, 44, activation_mode);
        set_karg(kernel_args, 45, uint8_t{255}); // sync_limit
        set_karg(kernel_args, 46, uint8_t{0});   // sync_period
        set_karg(kernel_args, 47, uint8_t{0});   // reserved8
        set_karg(kernel_args, 48, uint32_t{0});  // reserved9
        set_karg(kernel_args, 49, uint64_t{0});  // sync_addr
        set_karg(kernel_args, 50, uint64_t{0});  // acc_addr
        set_karg(kernel_args, 51, uint64_t{0});  // a_offset

        // Compute grid dimensions
        std::size_t grid_blocks = static_cast<std::size_t>(N) * G * ng;
        std::size_t global_size = grid_blocks * info.block_size;
        std::size_t local_size  = info.block_size;

        code_object_op cop;
        cop.code_object       = info.code_object;
        cop.symbol_name       = info.symbol_name;
        cop.global            = global_size;
        cop.local             = local_size;
        cop.expected_inputs   = inputs;
        cop.output            = out_shape;
        cop.output_arg        = static_cast<std::int64_t>(inputs.size() - 1);
        cop.kernel_args         = std::move(kernel_args);
        cop.runtime_arg_indices = std::move(runtime_arg_indices);

        return cop;
    }

    compiler_replace compile(context& ctx, instruction_ref ins, const operation& op) const
    {
        return compile_op(ctx, to_shapes(ins->inputs()), op.to_value());
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
