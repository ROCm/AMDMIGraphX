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
#include <migraphx/gpu/mlss_conv_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/register_op.hpp>
#ifdef MIGRAPHX_HAS_MLSS_HEADERS

namespace mlss_fp32 {
#include <archive/conv/mxn/Winograd/Base/gfx1201/fp32/ShaderTypes_GFX12_fp32_f2x3_stride1.llvm.cpp>
} // namespace mlss_fp32
namespace mlss_fp32_ostride2 {
#include <archive/conv/mxn/Winograd/Base/gfx1201/fp32/ShaderTypes_GFX12_fp32_f3x2_ostride2.llvm.cpp>
} // namespace mlss_fp32_ostride2
namespace mlss_fp16pk {
#include <archive/conv/mxn/Winograd/Rage/gfx1201/fp16/ShaderTypes_NAVI48_fp16pk_f2x3_stride1.llvm.cpp>
} // namespace mlss_fp16pk

#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_REGISTER_OP(mlss_conv_op);

mlss_conv_op mlss_conv_op::make_gfx12_fp32_f2x3_stride1()
{
    const auto& shader = mlss_fp32::GFX12_fp32_f2x3_stride1;
    mlss_conv_op op;
    op.code_object = value::binary(shader.m_binary.data(), shader.m_binary.size());
    op.symbol_name = "main";
    // 56 seems to give the best minimum latency on gfx1201
    op.n_groups    = 64;
    op.block_size  = 256;
    return op;
}

mlss_conv_op mlss_conv_op::make_gfx12_fp32_f3x2_ostride2()
{
    const auto& shader = mlss_fp32_ostride2::GFX12_fp32_f3x2_ostride2;
    mlss_conv_op op;
    op.code_object = value::binary(shader.m_binary.data(), shader.m_binary.size());
    op.symbol_name = "main";
    op.n_groups    = 64;
    op.block_size  = 256;
    return op;
}

mlss_conv_op mlss_conv_op::make_navi48_fp16pk_f2x3_stride1()
{
    const auto& shader = mlss_fp16pk::NAVI48_fp16pk_f2x3_stride1;
    mlss_conv_op op;
    op.code_object = value::binary(shader.m_binary.data(), shader.m_binary.size());
    op.symbol_name = "main";
    op.n_groups    = 64;
    op.block_size  = 384;
    return op;
}

// Pre-lowering: returns the stored output shape (no output buffer arg yet).
// Post-lowering: returns the shape of the last arg (the pre-allocated output buffer).
shape mlss_conv_op::compute_shape(std::vector<shape> inputs) const
{
    std::size_t expected = has_bias ? 3 : 2;
    if(inputs.size() > expected)
        return inputs.back();
    return output;
}

void mlss_conv_op::finalize(context&, const shape&, const std::vector<shape>&)
{
    assert(not code_object.empty());
    k = kernel(code_object, symbol_name);
}

argument mlss_conv_op::compute(context& ctx,
                                const shape&,
                                const std::vector<argument>& args) const
{
    // args layout (injected by find_mlss_conv / find_mlss_conv_bias):
    //   [0] input activation  e.g. float_type {1, 64, 128, 128}
    //   [1] weight literal    e.g. float_type {128, 64, 3, 3}
    //   [2] output buffer     e.g. float_type {1, 128, 128, 128}  (pre-allocated, returned)
    // When has_bias=true:
    //   [2] bias literal      e.g. float_type {128}
    //   [3] output buffer     e.g. float_type {1, 128, 128, 128}  (pre-allocated, returned)

    const auto& input  = args[0];
    const auto& weight = args[1];
    const auto& out_buf = args.back();

    const auto in_lens  = input.get_shape().lens();  // N, C, H, W
    const auto wt_lens  = weight.get_shape().lens(); // K, C, R, S
    const auto out_lens = out_buf.get_shape().lens(); // N, K, OH, OW

    // -----------------------------------------------------------------------
    // Geometry — matches kernel_execution_conv_fp32_f2x3_stride1_cg64_kg128.cpp
    // G=1 (single conv group), n_groups=64 (wavefront tile count).
    // -----------------------------------------------------------------------
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
    int32_t ng    = static_cast<int32_t>(n_groups);

    // Cap ng to prevent idle workgroups from writing out-of-bounds in the full model.
    //
    // The f2x3 Winograd kernel partitions work into tiles of:
    //   2 output rows  × 2 output cols  × 128 K channels  (F(2,3), "kg128" config)
    // Total tiles = N × G × ceil(OH/2) × ceil(OW/2) × ceil(Kg/128).
    //
    // With n_groups=64 and small spatial outputs, the number of dispatched workgroups
    // exceeds the total tile count.  Idle workgroups (tile_index >= total_tiles) still
    // perform out-of-bounds memory writes in the full model, corrupting adjacent buffers.
    // Setting ng = total_tiles ensures every dispatched workgroup has exactly one tile.
    //
    // Validation (Kg_per_wg = 128, per "kg128" kernel config name):
    //   {1,512,8,8}: total_tiles = 4*4*4 = 64 = ng → no idle wg, passes full model
    //   {1,256,8,8}: total_tiles = 4*4*2 = 32 < 64 → 32 idle wg, OOB writes, fails
    //   {1,256,4,4}: total_tiles = 2*2*2 = 8  < 64 → 56 idle wg, OOB writes, fails
    //
    // 9.1.2: kernarg n_groups MUST equal actual dispatch count.
    {
        const int32_t kg_per_workgroup = 128;
        int32_t k_groups    = (Kg + kg_per_workgroup - 1) / kg_per_workgroup;
        int32_t h_tiles     = (out_h + 1) / 2;
        int32_t w_tiles     = (out_w + 1) / 2;
        int32_t total_tiles = N * G * h_tiles * w_tiles * k_groups;
        if(ng > total_tiles)
            ng = total_tiles;
    }

    // flags64 encoding:
    //   no-bias path: bit10 = fast tile-index division
    //   bias path: bit7  = F_BIAS
    //              bit9  = F_NKCHR_STRIDES  (deprecated, recommended=1)
    //              bit14 = F_USE_ACTIVATION_MODE (deprecated, recommended=1)
    //              bit15 = F_USE_EXTENDED_FLAGS_64 (deprecated, recommended=1)
    uint64_t flags64 = has_bias
        ? ((uint64_t{1} << 7) | (uint64_t{1} << 9) | (uint64_t{1} << 14) | (uint64_t{1} << 15))
        : (uint64_t{1} << 10);

    // Device pointers as uint64_t (kernel ABI)
    uint64_t p_data   = reinterpret_cast<uint64_t>(input.data());
    uint64_t p_filter = reinterpret_cast<uint64_t>(weight.data());
    uint64_t p_output = reinterpret_cast<uint64_t>(out_buf.data());
    uint64_t p_bias   = has_bias ? reinterpret_cast<uint64_t>(args[2].data()) : 0;

    float alpha = activation_alpha;
    float beta  = 0.0f;

    // -----------------------------------------------------------------------
    // Strides (int32_t, NCHW)
    // -----------------------------------------------------------------------
    int32_t d_N_stride = G * Cg * H * W;
    int32_t d_C_stride = H * W;
    int32_t d_H_stride = W;
    int32_t d_G_stride = Cg * H * W;

    int32_t f_K_stride = Cg * R * S;
    int32_t f_C_stride = R * S;
    int32_t f_R_stride = S;
    int32_t f_G_stride = Kg * Cg * R * S;

    int32_t o_N_stride = G * Kg * out_h * out_w;
    int32_t o_K_stride = out_h * out_w;
    int32_t o_H_stride = out_w;
    int32_t o_G_stride = Kg * out_h * out_w;

    // Explicit zeros for padding/reserved slots in the kernarg buffer.
    uint32_t zero32 = 0;

    // pad_h and pad_w are set from the convolution op's padding attribute in apply()
    int32_t cur_pad_h = pad_h;
    int32_t cur_pad_w = pad_w;

    // -----------------------------------------------------------------------
    // Build kernarg list.
    // pack_args aligns each entry to its natural alignment, producing the
    // 0xe8-byte layout from kernel_execution_conv_fp32_f2x3_stride1_cg64_kg128.cpp:
    //
    //   0x00  N           int32
    //   0x04  Cg          int32
    //   0x08  H           int32
    //   0x0c  W           int32
    //   0x10  Kg          int32
    //   0x14  ng          int32   (.n_groups)
    //   0x18  flags64     uint64  (align-8 pad = 0, already aligned)
    //   0x20  p_data      uint64
    //   0x28  p_filter    uint64
    //   0x30  p_output    uint64
    //   0x38  zero64      uint64  (.reserved3)
    //   0x40  R           int32
    //   0x44  S           int32
    //   0x48  pad_h       int32
    //   0x4c  pad_w       int32
    //   0x50  out_h       int32
    //   0x54  out_w       int32
    //   0x58  p_bias      uint64  (align-8 pad = 0, already aligned)
    //   0x60  alpha       float
    //   0x64  beta        float
    //   0x68  zero32 × 8  uint32  (d/f/o/b offsets, unused)
    //   0x88  d_N_stride  int32
    //   0x8c  d_C_stride  int32
    //   0x90  d_H_stride  int32
    //   0x94  zero32      uint32  (reserved4)
    //   0x98  f_K_stride  int32
    //   0x9c  f_C_stride  int32
    //   0xa0  f_R_stride  int32
    //   0xa4  zero32      uint32  (reserved5)
    //   0xa8  o_N_stride  int32
    //   0xac  o_K_stride  int32
    //   0xb0  o_H_stride  int32
    //   0xb4  zero32      uint32  (reserved6)
    //   0xb8  G           int32
    //   0xbc  d_G_stride  int32
    //   0xc0  f_G_stride  int32
    //   0xc4  o_G_stride  int32
    // When has_bias (F_BIAS set) the kernel reads 8 more fields (0xc8–0xe7):
    //   0xc8  activation_mode  uint8  (0 = none, 4 = ReLU)
    //   0xc9  sync_limit       uint8  (255 = DEFAULT_SYNC_LIMIT)
    //   0xca  sync_period      uint8  (0)
    //   0xcb  reserved8        uint8  (0)
    //   0xcc  reserved9        uint32 (0)
    //   0xd0  sync_addr        uint64 (pointer to d_sync, zeroed)
    //   0xd8  acc_addr         uint64 (pointer to d_acc, zeroed)
    //   0xe0  a_offset         uint64 (0)
    // -----------------------------------------------------------------------
    std::vector<kernel_argument> kargs;

    kargs.emplace_back(N);
    kargs.emplace_back(Cg);
    kargs.emplace_back(H);
    kargs.emplace_back(W);
    kargs.emplace_back(Kg);
    kargs.emplace_back(ng);
    kargs.emplace_back(flags64);     // uint64, align-8 → offset 0x18
    kargs.emplace_back(p_data);      // uint64 → offset 0x20
    kargs.emplace_back(p_filter);    // uint64 → offset 0x28
    kargs.emplace_back(p_output);    // uint64 → offset 0x30
    kargs.emplace_back(uint64_t{0}); // reserved3 → offset 0x38
    kargs.emplace_back(R);
    kargs.emplace_back(S);
    kargs.emplace_back(cur_pad_h);
    kargs.emplace_back(cur_pad_w);
    kargs.emplace_back(out_h);
    kargs.emplace_back(out_w);
    kargs.emplace_back(p_bias);      // uint64, align-8 → offset 0x58
    kargs.emplace_back(alpha);
    kargs.emplace_back(beta);
    // d/f/o/b offsets (8 × uint32, unused) → offsets 0x68–0x87
    kargs.emplace_back(zero32); kargs.emplace_back(zero32);
    kargs.emplace_back(zero32); kargs.emplace_back(zero32);
    kargs.emplace_back(zero32); kargs.emplace_back(zero32);
    kargs.emplace_back(zero32); kargs.emplace_back(zero32);
    kargs.emplace_back(d_N_stride);
    kargs.emplace_back(d_C_stride);
    kargs.emplace_back(d_H_stride);
    kargs.emplace_back(zero32);      // reserved4 → offset 0x94
    kargs.emplace_back(f_K_stride);
    kargs.emplace_back(f_C_stride);
    kargs.emplace_back(f_R_stride);
    kargs.emplace_back(zero32);      // reserved5 → offset 0xa4
    kargs.emplace_back(o_N_stride);
    kargs.emplace_back(o_K_stride);
    kargs.emplace_back(o_H_stride);
    kargs.emplace_back(zero32);      // reserved6 → offset 0xb4
    kargs.emplace_back(G);
    kargs.emplace_back(d_G_stride);
    kargs.emplace_back(f_G_stride);
    kargs.emplace_back(o_G_stride);
    uint8_t act_mode      = static_cast<uint8_t>(activation_mode);
    uint8_t sync_limit_v  = 255; // DEFAULT_SYNC_LIMIT (conv_base.hpp)
    uint8_t sync_period_v = 0;
    uint8_t reserved8_v   = 0;
    uint64_t sync_addr_v  = 0;
    uint64_t acc_addr_v   = 0;
    uint64_t a_offset_v   = 0;

    kargs.emplace_back(act_mode);      // 0xc8: activation_mode
    kargs.emplace_back(sync_limit_v);  // 0xc9: sync_limit
    kargs.emplace_back(sync_period_v); // 0xca: sync_period
    kargs.emplace_back(reserved8_v);   // 0xcb: reserved8
    kargs.emplace_back(zero32);        // 0xcc: reserved9
    kargs.emplace_back(sync_addr_v);   // 0xd0: sync_addr
    kargs.emplace_back(acc_addr_v);    // 0xd8: acc_addr
    kargs.emplace_back(a_offset_v);    // 0xe0: a_offset

    hipStream_t stream = ctx.get_stream().get();

    // -----------------------------------------------------------------------
    // Launch: grid = N * G * ng workgroups of block_size threads each.
    // ng may have been capped below n_groups for small spatial outputs (see above).
    // -----------------------------------------------------------------------
    std::size_t grid_blocks = static_cast<std::size_t>(N) * G * ng;

    auto [start, stop] = ctx.get_perf_events();
    k.launch(stream, grid_blocks * block_size, block_size, kargs, start, stop);

    return args.back();
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_HAS_MLSS_HEADERS
