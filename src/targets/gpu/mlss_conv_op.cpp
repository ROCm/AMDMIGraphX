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
#include <migraphx/check_shapes.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/register_op.hpp>
#ifdef MIGRAPHX_HAS_MLSS_HEADERS

#ifdef MIGRAPHX_USE_AMDMLSS
#include <amdmlss/amdmlss_api.h>
#include <iostream>
#else
#include "modules/shaders/src/operators/impl/conv/mxn/Winograd/Base/gfx1201/fp32/shadersBinNonReloc.hpp"
#endif

#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_REGISTER_OP(mlss_conv_op);

mlss_conv_op mlss_conv_op::make_gfx12_fp32_f2x3_stride1(
    const context& ctx,
    const std::vector<std::size_t>& act_lens,
    const std::vector<std::size_t>& wt_lens,
    const std::vector<std::size_t>& out_lens,
    const std::vector<std::size_t>& padding,
    const std::vector<std::size_t>& stride,
    const std::vector<std::size_t>& dilation,
    std::size_t group,
    bool has_bias_flag,
    uint8_t act_mode,
    shape::type_t dtype)
{
    mlss_conv_op op;
    op.block_size = 256;

#ifdef MIGRAPHX_USE_AMDMLSS
    const std::string gfx_name = ctx.get_current_device().get_gfx_name();
    MLSScontext mlss_ctx        = 0;
    MLSSstring op_name          = const_cast<MLSSstring>(MLSS_CONV);
    if(mlssCreateContext(&mlss_ctx, const_cast<MLSSstring>(gfx_name.c_str()), op_name) !=
       MLSS_SUCCESS)
    {
        MIGRAPHX_THROW("mlss_conv_op: mlssCreateContext failed for " + gfx_name);
    }

    // Dimensions: act=[N,C,H,W], wt=[K,C/g,R,S], out=[N,K,outH,outW]
    std::uint32_t n    = static_cast<std::uint32_t>(act_lens[0]);
    std::uint32_t c    = static_cast<std::uint32_t>(act_lens[1]);
    std::uint32_t h    = static_cast<std::uint32_t>(act_lens[2]);
    std::uint32_t w    = static_cast<std::uint32_t>(act_lens[3]);
    std::uint32_t k    = static_cast<std::uint32_t>(wt_lens[0]);
    std::uint32_t r    = static_cast<std::uint32_t>(wt_lens[2]);
    std::uint32_t s    = static_cast<std::uint32_t>(wt_lens[3]);
    std::uint32_t outH = static_cast<std::uint32_t>(out_lens[2]);
    std::uint32_t outW = static_cast<std::uint32_t>(out_lens[3]);

    std::uint32_t dilationX = dilation.size() > 1 ? static_cast<std::uint32_t>(dilation[1]) : 1;
    std::uint32_t dilationY = dilation.size() > 0 ? static_cast<std::uint32_t>(dilation[0]) : 1;

    std::uint32_t startPadY = padding.size() > 0 ? static_cast<std::uint32_t>(padding[0]) : 0;
    std::uint32_t startPadX = padding.size() > 1 ? static_cast<std::uint32_t>(padding[1]) : 0;
    std::uint32_t endPadY   = padding.size() > 2 ? static_cast<std::uint32_t>(padding[2]) : 0;
    std::uint32_t endPadX   = padding.size() > 3 ? static_cast<std::uint32_t>(padding[3]) : 0;
    std::uint32_t outPadX   = 0;
    std::uint32_t outPadY   = 0;

    std::uint32_t convStrideY   = stride.size() > 0 ? static_cast<std::uint32_t>(stride[0]) : 1;
    std::uint32_t convStrideX   = stride.size() > 1 ? static_cast<std::uint32_t>(stride[1]) : 1;
    std::uint32_t inputStrideX  = 1;
    std::uint32_t inputStrideY  = 1;
    std::uint32_t filterStrideX = 1;
    std::uint32_t filterStrideY = 1;

    std::uint32_t groups           = static_cast<std::uint32_t>(group);
    MLSSbool      mlss_has_bias    = has_bias_flag ? true : false;
    MLSSbool      crossCorrelation = false;
    MLSSbool      backward         = false;

    // Tensor strides (NCHW, groups=1)
    std::uint32_t dNStride = c * h * w;
    std::uint32_t dHStride = w;
    std::uint32_t dCStride = h * w;
    std::uint32_t fKStride = c * r * s;
    std::uint32_t fCStride = r * s;
    std::uint32_t fRStride = s;
    std::uint32_t fSStride = 1;
    std::uint32_t oNStride = k * outH * outW;
    std::uint32_t oHStride = outW;
    std::uint32_t oKStride = outH * outW;
    std::uint32_t dOffset  = 0;
    std::uint32_t oOffset  = 0;
    std::uint32_t fOffset  = 0;
    std::uint32_t bOffset  = 0;

    MLSSenum dataType  = (dtype == shape::half_type) ? MLSS_FLOAT16 : MLSS_FLOAT32;
    MLSSenum precision = static_cast<MLSSenum>(MLSS_PRECISION_FLOAT16_ADD_FLOAT32);

    // Map MIGraphX mlss_activation_mode (identity=0, leaky_relu=1, relu=4)
    // to AMDMLSS MLSSActivationFunctionFlag (identity=3, leaky_relu=4, relu=9)
    MLSSenum activation;
    switch(static_cast<mlss_activation_mode>(act_mode))
    {
    case mlss_activation_mode::identity:   activation = MLSS_ACTIVATION_IDENTITY;   break;
    case mlss_activation_mode::leaky_relu: activation = MLSS_ACTIVATION_LEAKY_RELU; break;
    case mlss_activation_mode::sigmoid:    activation = MLSS_ACTIVATION_SIGMOID;    break;
    case mlss_activation_mode::scaled_tanh:activation = MLSS_ACTIVATION_SCALED_TANH;break;
    case mlss_activation_mode::relu:       activation = MLSS_ACTIVATION_RELU;       break;
    }

    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_W, &w);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_H, &h);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_C, &c);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_N, &n);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_K, &k);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_S, &s);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_R, &r);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_OUTW, &outW);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_OUTH, &outH);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_DILATIONX, &dilationX);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_DILATIONY, &dilationY);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_STARTPADX, &startPadX);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_STARTPADY, &startPadY);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_ENDPADX, &endPadX);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_ENDPADY, &endPadY);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_OUTPADX, &outPadX);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_OUTPADY, &outPadY);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_CONVSTRIDEX, &convStrideX);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_CONVSTRIDEY, &convStrideY);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_INPUTSTRIDEX, &inputStrideX);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_INPUTSTRIDEY, &inputStrideY);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_FILTERSTRIDEX, &filterStrideX);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_FILTERSTRIDEY, &filterStrideY);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_GROUPS, &groups);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_HASBIAS, &mlss_has_bias);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_CROSSCORRELATION, &crossCorrelation);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_BACKWARD, &backward);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_DNSTRIDE, &dNStride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_DHSTRIDE, &dHStride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_DCSTRIDE, &dCStride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_FKSTRIDE, &fKStride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_FCSTRIDE, &fCStride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_FRSTRIDE, &fRStride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_FSSTRIDE, &fSStride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_ONSTRIDE, &oNStride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_OHSTRIDE, &oHStride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_OKSTRIDE, &oKStride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_DOFFSET, &dOffset);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_OOFFSET, &oOffset);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_FOFFSET, &fOffset);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_BOFFSET, &bOffset);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_DATATYPE, &dataType);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_PRECISION, &precision);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_ACTIVATION, &activation);

    MLSSstatus* p_statuses = nullptr;
    MLSSsize n_statuses    = 0;
    MLSSstatus caps_status = mlssGetCaps(mlss_ctx, &p_statuses, &n_statuses);
    if(caps_status != MLSS_SUCCESS)
    {
        std::cout << "[mlss_conv_op] getCaps failed (status=" << caps_status << ") for "
                  << gfx_name << " ["
                  << n << "x" << c << "x" << h << "x" << w
                  << " * " << k << "x" << (c / groups) << "x" << r << "x" << s
                  << " -> " << n << "x" << k << "x" << outH << "x" << outW
                  << "]  groups=" << groups
                  << " dil=" << dilationX << "x" << dilationY
                  << " stride=" << convStrideX << "x" << convStrideY
                  << " pad=" << startPadX << "," << startPadY << "," << endPadX << "," << endPadY
                  << " bias=" << mlss_has_bias
                  << " act=" << activation
                  << " dtype=" << dataType
                  << " prec=" << precision << "\n";
        mlssPrintParameters(mlss_ctx, op_name);
        return op;
    }

    MLSSbinary* binaries  = nullptr;
    MLSSsize num_binaries = 0;
    if(mlssGetBinaries(mlss_ctx, &binaries, &num_binaries) != MLSS_SUCCESS || num_binaries == 0)
        return op; // no binaries available — return empty op

    // Find first non-relocatable binary
    const MLSSbinary* bin = nullptr;
    for(MLSSsize i = 0; i < num_binaries; ++i)
    {
        if(not binaries[i].m_isRelocatable)
        {
            bin = &binaries[i];
            break;
        }
    }
    if(bin == nullptr)
        return op; // no non-relocatable binary — return empty op

    const auto* raw = static_cast<const char*>(bin->m_binaries);
    op.code_object  = value::binary(raw, bin->m_binarySize);
    op.symbol_name  = (bin->m_pKernelName != nullptr) ? bin->m_pKernelName : "main";

    // Use the producer-chosen grid to derive n_groups.
    // Grid = N * G * n_groups, with N and G from the shape.
    std::size_t grid_x = bin->m_grid.m_x;
    op.n_groups = grid_x / (static_cast<std::size_t>(n) * static_cast<std::size_t>(groups));
    if(op.n_groups == 0)
        op.n_groups = 64;

    std::cout << "[mlss_conv_op] API binary: size=" << bin->m_binarySize
              << "  kernel=" << op.symbol_name << " ["
                << n << "x" << c << "x" << h << "x" << w
                << " * " << k << "x" << (c / groups) << "x" << r << "x" << s
                << " -> " << n << "x" << k << "x" << outH << "x" << outW
                << "]  groups=" << groups
              << "  grid={" << bin->m_grid.m_x << "," << bin->m_grid.m_y << "," << bin->m_grid.m_z
              << "}  n_groups=" << op.n_groups << "\n";

#else
    (void)ctx; (void)act_lens; (void)wt_lens; (void)out_lens;
    (void)padding; (void)stride; (void)dilation; (void)group;
    (void)has_bias_flag; (void)act_mode; (void)dtype;
    const auto& shader = mlss::conv::mxn::winograd::base::fp32::gfx1201::ConvWinogradElf_Gfx12_F2x3_Fp32Stride1_NonReloc;
    op.code_object = value::binary(shader.m_binary.data(), shader.m_binary.size());
    op.symbol_name = "main";
    op.n_groups    = 64;
#endif // MIGRAPHX_USE_AMDMLSS

    return op;
}

// mlss_conv_op mlss_conv_op::make_gfx12_fp32_f3x2_ostride2()
// {
//     const auto& shader = mlss_fp32_ostride2::GFX12_fp32_f3x2_ostride2;
//     mlss_conv_op op;
//     op.code_object = value::binary(shader.m_binary.data(), shader.m_binary.size());
//     op.symbol_name = "main";
//     op.n_groups    = 64;
//     op.block_size  = 256;
//     return op;
// }

// mlss_conv_op mlss_conv_op::make_navi48_fp16pk_f2x3_stride1()
// {
//     const auto& shader = mlss_fp16pk::NAVI48_fp16pk_f2x3_stride1;
//     mlss_conv_op op;
//     op.code_object = value::binary(shader.m_binary.data(), shader.m_binary.size());
//     op.symbol_name = "main";
//     op.n_groups    = 64;
//     op.block_size  = 384;
//     return op;
// }

// Pre-lowering: returns the stored output shape (no output buffer arg yet).
// Post-lowering: returns the shape of the last arg (the pre-allocated output buffer).
shape mlss_conv_op::compute_shape(std::vector<shape> inputs) const
{
    check_shapes{inputs, *this}.standard();
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
