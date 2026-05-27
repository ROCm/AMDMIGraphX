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
#include <migraphx/errors.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_REGISTER_OP(mlss_conv_op);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#ifdef MIGRAPHX_USE_AMDMLSS
#include <amdmlss/amdmlss_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

mlss_conv_binary_info query_mlss_conv_binary(
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
    mlss_conv_binary_info info;

    mlssSetVerboseLevel(MLSS_VERBOSE_NONE);

    const std::string gfx_name = ctx.get_current_device().get_gfx_name();
    MLSScontext mlss_ctx        = 0;
    MLSSstring op_name          = const_cast<MLSSstring>(MLSS_CONV);
    if(mlssCreateContext(&mlss_ctx, const_cast<MLSSstring>(gfx_name.c_str()), op_name) !=
       MLSS_SUCCESS)
    {
        MIGRAPHX_THROW("mlss_conv: mlssCreateContext failed for " + gfx_name);
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

    // Tensor strides (NCHW)
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

    // Map MIGraphX mlss_activation_mode to AMDMLSS MLSSActivationFunctionFlag
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
    if(mlssGetCaps(mlss_ctx, &p_statuses, &n_statuses) != MLSS_SUCCESS)
        return info;

    MLSSbinary* binaries  = nullptr;
    MLSSsize num_binaries = 0;
    if(mlssGetBinaries(mlss_ctx, &binaries, &num_binaries) != MLSS_SUCCESS || num_binaries == 0)
        return info;

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
        return info;

    const auto* raw = static_cast<const char*>(bin->m_binaries);
    info.code_object = value::binary(raw, bin->m_binarySize);
    info.symbol_name = (bin->m_pKernelName != nullptr) ? bin->m_pKernelName : "main";

    // Derive n_groups from the producer-chosen grid
    std::size_t grid_x = bin->m_grid.m_x;
    info.n_groups = grid_x / (static_cast<std::size_t>(n) * static_cast<std::size_t>(groups));
    if(info.n_groups == 0)
        info.n_groups = 64;

    // Block size from binary metadata, fallback by dtype
    info.block_size = (bin->m_blocks.m_x > 1) ? bin->m_blocks.m_x
                    : (dtype == shape::half_type) ? 384 : 256;

    return info;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_USE_AMDMLSS
