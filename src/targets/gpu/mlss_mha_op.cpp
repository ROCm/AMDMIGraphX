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
#include <migraphx/gpu/mlss_mha_op.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/register_op.hpp>
#ifdef MIGRAPHX_USE_AMDMLSS
#include <amdmlss/amdmlss_api.h>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_REGISTER_OP(mlss_mha_op);

mlss_mha_op mlss_mha_op::make_gfx1201_fp16_packed_qkv(const context& ctx,
                                                      const std::vector<std::size_t>& query_lens,
                                                      float scale,
                                                      std::size_t global,
                                                      std::size_t local)
{
    mlss_mha_op op;
    op.global = global;
    op.local  = local;
    op.scale  = scale;

    mlssSetVerboseLevel(MLSS_VERBOSE_NONE);

    const std::string gfx_name = ctx.get_current_device().get_gfx_name();
    MLSScontext mlss_ctx       = 0;
    MLSSstring op_name         = const_cast<MLSSstring>(MLSS_MHA);
    if(mlssCreateContext(&mlss_ctx, const_cast<MLSSstring>(gfx_name.c_str()), op_name) !=
       MLSS_SUCCESS)
        return op;

    std::uint32_t batch   = static_cast<std::uint32_t>(query_lens[0]);
    std::uint32_t heads   = static_cast<std::uint32_t>(query_lens[1]);
    std::uint32_t q_seq   = static_cast<std::uint32_t>(query_lens[2]);
    std::uint32_t kv_seq  = q_seq;
    std::uint32_t h_dim   = static_cast<std::uint32_t>(query_lens[3]);
    std::uint32_t kv_dim  = 0;
    std::uint32_t packing = MLSS_ATTR_CONFIG_MHA_PACKING_PACKED_QKV;
    float scale_val       = scale;
    MLSSenum dtype        = MLSS_FLOAT16;

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
    if(mlssGetCaps(mlss_ctx, &p_statuses, &n_statuses) != MLSS_SUCCESS)
        return op;

    MLSSbinary* binaries  = nullptr;
    MLSSsize num_binaries = 0;
    if(mlssGetBinaries(mlss_ctx, &binaries, &num_binaries) != MLSS_SUCCESS || num_binaries == 0)
        return op;

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
        return op;

    const auto* raw = static_cast<const char*>(bin->m_binaries);
    op.code_object  = value::binary(raw, bin->m_binarySize);
    op.symbol_name  = (bin->m_pKernelName != nullptr) ? bin->m_pKernelName : "main";

    return op;
}

shape mlss_mha_op::compute_shape(std::vector<shape> inputs) const { return inputs.back(); }

void mlss_mha_op::finalize(context&, const shape&, const std::vector<shape>&)
{
    assert(not code_object.empty());
    k = kernel(code_object, symbol_name);
}

argument mlss_mha_op::compute(context& ctx, const shape&, const std::vector<argument>& args) const
{
    // args layout:
    //   [0] packed QKV  [B, S, H, 3*D]
    //   [1] K  (same buffer, different logical offset — packed)
    //   [2] V  (same buffer, different logical offset — packed)
    //   [3] output buffer  [B, H, S, D]
    // scale is packed into op.scale and passed directly to the kernel
    constexpr int output_arg_idx = 3;

    auto query          = args[0];
    auto outval         = args[output_arg_idx];
    auto query_lens     = query.get_shape().lens();
    auto outval_strides = outval.get_shape().strides();

    int batch_size      = static_cast<int>(query_lens[0]);
    int head_num        = static_cast<int>(query_lens[1]);
    int sequence_length = static_cast<int>(query_lens[2]);
    int head_dim        = static_cast<int>(query_lens[3]);

    // QKV strides for the [B, S, H, 3*D] seq-major packed layout:
    //   d0 = S * H * 3*D   (batch stride)
    //   d1 = 3*D           (head stride)
    //   d2 = H * 3*D       (sequence stride)
    //   d3 = 1             (element stride)
    constexpr int qkv_components = 3;
    uint32_t stride_d0 =
        static_cast<uint32_t>(sequence_length * head_num * qkv_components * head_dim);
    uint32_t stride_d1 = static_cast<uint32_t>(qkv_components * head_dim);
    uint32_t stride_d2 = static_cast<uint32_t>(head_num * qkv_components * head_dim);
    uint32_t stride_d3 = 1u;

    // kernel_argument stores void* — all values must be named locals (non-const) to
    // allow &x → void* conversion, and must outlive k.launch().
    hipDeviceptr_t d_q_in   = query.data();
    hipDeviceptr_t d_out_in = outval.data();

    float scale_val = scale; // copy: compute() is const, so `scale` would be const float

    uint32_t output_stride_d0 = static_cast<uint32_t>(outval_strides[0]);
    uint32_t output_stride_d1 = static_cast<uint32_t>(outval_strides[1]);
    uint32_t output_stride_d2 = static_cast<uint32_t>(outval_strides[2]);
    uint32_t output_stride_d3 = static_cast<uint32_t>(outval_strides[3]);

    std::vector<kernel_argument> kargs;
    kargs.emplace_back(d_q_in);
    kargs.emplace_back(d_out_in);
    kargs.push_back(batch_size);
    kargs.push_back(sequence_length);
    kargs.push_back(head_num);
    kargs.push_back(head_dim);
    kargs.push_back(scale_val);
    // Q strides
    kargs.push_back(stride_d0);
    kargs.push_back(stride_d1);
    kargs.push_back(stride_d2);
    kargs.push_back(stride_d3);
    // K strides (same layout as Q)
    kargs.push_back(stride_d0);
    kargs.push_back(stride_d1);
    kargs.push_back(stride_d2);
    kargs.push_back(stride_d3);
    // V strides (d2/d3 swapped per MLSS kernel convention)
    kargs.push_back(stride_d0);
    kargs.push_back(stride_d1);
    kargs.push_back(stride_d3);
    kargs.push_back(stride_d2);
    // output strides
    kargs.push_back(output_stride_d0);
    kargs.push_back(output_stride_d1);
    kargs.push_back(output_stride_d2);
    kargs.push_back(output_stride_d3);

    auto [start, stop] = ctx.get_perf_events();
    k.launch(ctx.get_stream().get(), global, local, kargs, start, stop);

    return args[output_arg_idx];
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_USE_AMDMLSS
