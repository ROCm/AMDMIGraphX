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
#include <migraphx/check_shapes.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/make_op.hpp>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Intermediate op inserted by fuse_mlss. Carries conv metadata for the JIT
// compiler (jit/mlss_conv.cpp) which converts it into a code_object_op.
struct mlss_conv_op
{
    operation conv_op = make_op("convolution");
    bool has_bias     = false;
    // The cast is needed (enum class → uint8_t isn't implicit)
    // cppcheck-suppress migraphx-RedundantCast
    uint8_t activation_mode = static_cast<uint8_t>(mlss_activation_mode::identity);
    float activation_alpha  = 0.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.conv_op, "conv_op"),
                    f(self.has_bias, "has_bias"),
                    f(self.activation_mode, "activation_mode"),
                    f(self.activation_alpha, "activation_alpha"));
    }

    std::string name() const { return "gpu::mlss_conv"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        // Inputs are [activation, weight] plus an optional [bias] when has_bias.
        const std::size_t expected = has_bias ? 3 : 2;
        check_shapes{inputs, *this}.has(expected).same_type();
        // The bias does not affect the output shape, so only the
        // [activation, weight] pair is forwarded to the conv op.
        return conv_op.compute_shape({inputs.at(0), inputs.at(1)});
    }
};

MIGRAPHX_REGISTER_OP(mlss_conv_op);

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#ifdef MIGRAPHX_USE_AMDMLSS
#include <amdmlss/amdmlss_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// NOLINTNEXTLINE(readability-function-size)
mlss_conv_binary_info query_mlss_conv_binary(const context& ctx,
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

    // mlssCreateContext takes non-const MLSSstring (char*)
    std::string gfx_name    = ctx.get_current_device().get_gfx_name();
    std::string op_name_str = MLSS_CONV;
    MLSSstring op_name      = op_name_str.data();
    MLSScontext mlss_ctx    = 0;
    if(mlssCreateContext(&mlss_ctx, gfx_name.data(), op_name) != MLSS_SUCCESS)
    {
        MIGRAPHX_THROW("mlss_conv: mlssCreateContext failed for " + gfx_name);
    }

    std::uint32_t n     = act_lens[0];
    std::uint32_t c     = act_lens[1];
    std::uint32_t h     = act_lens[2];
    std::uint32_t w     = act_lens[3];
    std::uint32_t k     = wt_lens[0];
    std::uint32_t r     = wt_lens[2];
    std::uint32_t s     = wt_lens[3];
    std::uint32_t out_h = out_lens[2];
    std::uint32_t out_w = out_lens[3];

    std::uint32_t dilation_x = dilation.size() > 1 ? static_cast<std::uint32_t>(dilation[1]) : 1;
    std::uint32_t dilation_y = dilation.size() > 0 ? static_cast<std::uint32_t>(dilation[0]) : 1;

    std::uint32_t start_pad_y = padding.size() > 0 ? static_cast<std::uint32_t>(padding[0]) : 0;
    std::uint32_t start_pad_x = padding.size() > 1 ? static_cast<std::uint32_t>(padding[1]) : 0;
    std::uint32_t end_pad_y   = padding.size() > 2 ? static_cast<std::uint32_t>(padding[2]) : 0;
    std::uint32_t end_pad_x   = padding.size() > 3 ? static_cast<std::uint32_t>(padding[3]) : 0;
    std::uint32_t out_pad_x   = 0;
    std::uint32_t out_pad_y   = 0;

    std::uint32_t conv_stride_y   = stride.size() > 0 ? static_cast<std::uint32_t>(stride[0]) : 1;
    std::uint32_t conv_stride_x   = stride.size() > 1 ? static_cast<std::uint32_t>(stride[1]) : 1;
    std::uint32_t input_stride_x  = 1;
    std::uint32_t input_stride_y  = 1;
    std::uint32_t filter_stride_x = 1;
    std::uint32_t filter_stride_y = 1;

    std::uint32_t groups       = group;
    MLSSbool mlss_has_bias     = has_bias_flag;
    MLSSbool cross_correlation = false;
    MLSSbool backward          = false;

    // Tensor strides (NCHW)
    std::uint32_t d_n_stride = c * h * w;
    std::uint32_t d_h_stride = w;
    std::uint32_t d_c_stride = h * w;
    std::uint32_t f_k_stride = c * r * s;
    std::uint32_t f_c_stride = r * s;
    std::uint32_t f_r_stride = s;
    std::uint32_t f_s_stride = 1;
    std::uint32_t o_n_stride = k * out_h * out_w;
    std::uint32_t o_h_stride = out_w;
    std::uint32_t o_k_stride = out_h * out_w;
    std::uint32_t d_offset   = 0;
    std::uint32_t o_offset   = 0;
    std::uint32_t f_offset   = 0;
    std::uint32_t b_offset   = 0;

    MLSSenum data_type = (dtype == shape::half_type) ? MLSS_FLOAT16 : MLSS_FLOAT32;
    MLSSenum precision = MLSS_PRECISION_FLOAT16_ADD_FLOAT32;

    // Map MIGraphX mlss_activation_mode to AMDMLSS MLSSActivationFunctionFlag.
    // Validate against the enum's `last` sentinel before casting so the switch
    // covers every enumerator (avoids covered-switch-default). Adding a new
    // mode requires bumping `last` in the enum, which keeps this guard in sync.
    if(act_mode > static_cast<uint8_t>(mlss_activation_mode::last))
        MIGRAPHX_THROW("mlss_conv: unknown activation mode " +
                       std::to_string(static_cast<int>(act_mode)));
    MLSSenum activation = MLSS_ACTIVATION_IDENTITY;
    switch(static_cast<mlss_activation_mode>(act_mode))
    {
    case mlss_activation_mode::identity: activation = MLSS_ACTIVATION_IDENTITY; break;
    case mlss_activation_mode::leaky_relu: activation = MLSS_ACTIVATION_LEAKY_RELU; break;
    case mlss_activation_mode::sigmoid: activation = MLSS_ACTIVATION_SIGMOID; break;
    case mlss_activation_mode::scaled_tanh: activation = MLSS_ACTIVATION_SCALED_TANH; break;
    case mlss_activation_mode::relu: activation = MLSS_ACTIVATION_RELU; break;
    }

    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_W, &w);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_H, &h);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_C, &c);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_N, &n);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_K, &k);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_S, &s);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_R, &r);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_OUTW, &out_w);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_OUTH, &out_h);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_DILATIONX, &dilation_x);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_DILATIONY, &dilation_y);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_STARTPADX, &start_pad_x);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_STARTPADY, &start_pad_y);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_ENDPADX, &end_pad_x);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_ENDPADY, &end_pad_y);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_OUTPADX, &out_pad_x);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_OUTPADY, &out_pad_y);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_CONVSTRIDEX, &conv_stride_x);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_CONVSTRIDEY, &conv_stride_y);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_INPUTSTRIDEX, &input_stride_x);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_INPUTSTRIDEY, &input_stride_y);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_FILTERSTRIDEX, &filter_stride_x);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_FILTERSTRIDEY, &filter_stride_y);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_GROUPS, &groups);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_HASBIAS, &mlss_has_bias);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_CROSSCORRELATION, &cross_correlation);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_BACKWARD, &backward);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_DNSTRIDE, &d_n_stride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_DHSTRIDE, &d_h_stride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_DCSTRIDE, &d_c_stride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_FKSTRIDE, &f_k_stride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_FCSTRIDE, &f_c_stride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_FRSTRIDE, &f_r_stride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_FSSTRIDE, &f_s_stride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_ONSTRIDE, &o_n_stride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_OHSTRIDE, &o_h_stride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_OKSTRIDE, &o_k_stride);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_DOFFSET, &d_offset);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_OOFFSET, &o_offset);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_FOFFSET, &f_offset);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_BOFFSET, &b_offset);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_DATATYPE, &data_type);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_PRECISION, &precision);
    mlssSetParameterByEnum(&mlss_ctx, op_name, MLSS_ATTR_CONV_ACTIVATION, &activation);

    MLSSstatus* p_statuses = nullptr;
    MLSSsize n_statuses    = 0;
    if(mlssGetCaps(mlss_ctx, &p_statuses, &n_statuses) != MLSS_SUCCESS)
        return info;

    // Each entry indicates whether the corresponding op configuration is
    // supported. Reject if any is non-success (e.g. MLSS_ERROR_SHADER_*)
    if(p_statuses == nullptr or n_statuses == 0)
        return info;
    for(MLSSsize i = 0; i < n_statuses; ++i)
    {
        if(p_statuses[i] != MLSS_SUCCESS)
            return info;
    }

    MLSSbinary* binaries  = nullptr;
    MLSSsize num_binaries = 0;
    if(mlssGetBinaries(mlss_ctx, &binaries, &num_binaries) != MLSS_SUCCESS or num_binaries == 0)
        return info;

    // Find first non-relocatable binary whose entry point is named "main"
    const MLSSbinary* bin = nullptr;
    for(MLSSsize i = 0; i < num_binaries; ++i)
    {
        if(not binaries[i].m_isRelocatable and binaries[i].m_pKernelName != nullptr and
           std::string(binaries[i].m_pKernelName) == "main")
        {
            bin = &binaries[i];
            break;
        }
    }
    if(bin == nullptr)
        return info;

    // cppcheck-suppress migraphx-RedundantCast
    const auto* raw = static_cast<const char*>(bin->m_binaries);

    // Verify the binary is actually loadable as a hip module before using it
    {
        hipModule_t raw_m = nullptr;
        if(hipModuleLoadData(&raw_m, raw) != hipSuccess)
            return info;
        (void)hipModuleUnload(raw_m);
    }

    info.code_object = value::binary(raw, bin->m_binarySize);
    info.symbol_name = (bin->m_pKernelName != nullptr) ? bin->m_pKernelName : "main";

    // Derive n_groups from the producer-chosen grid
    std::size_t grid_x = bin->m_grid.m_x;
    info.n_groups      = grid_x / (static_cast<std::size_t>(n) * static_cast<std::size_t>(groups));
    if(info.n_groups == 0)
        info.n_groups = 64;

    // Block size from binary metadata, fallback by dtype
    info.block_size = (bin->m_blocks.m_x > 1)       ? bin->m_blocks.m_x
                      : (dtype == shape::half_type) ? 384
                                                    : 256;

    return info;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_USE_AMDMLSS
