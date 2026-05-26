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
#ifndef MIGRAPHX_GUARD_AMDMIGRAPHX_MLSS_CONV_OP_HPP
#define MIGRAPHX_GUARD_AMDMIGRAPHX_MLSS_CONV_OP_HPP

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/gpu/kernel.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

enum class mlss_activation_mode : uint8_t
{
    identity    = 0,
    leaky_relu  = 1,
    sigmoid     = 2,
    scaled_tanh = 3,
    relu        = 4,
};

struct mlss_conv_op
{
    value::binary code_object{};
    std::string symbol_name{};
    // n_groups: wavefront tile count (used to compute grid size)
    std::size_t n_groups   = 64;
    // block_size: workgroup thread count — fp32=256, fp16pk=384
    std::size_t block_size = 256;
    // pad_h, pad_w: convolution padding along H and W axes
    int32_t pad_h = 0;
    int32_t pad_w = 0;
    // has_bias: when true args layout is [input, weight, bias, output]
    // and the kernel is launched with F_BIAS (bit 7) set in flags64.
    bool has_bias = false;
    // activation_mode: kernel activation applied after bias add.
    uint8_t activation_mode = static_cast<uint8_t>(mlss_activation_mode::identity);
    // activation_alpha: parameter for parameterized activations (e.g. leaky_relu slope).
    // Passed to the kernel via the alpha field (offset 0x60 in the kernarg buffer).
    float activation_alpha = 0.0f;
    // output: expected output shape, set during fusion.  Used by compute_shape
    // before the lowering pass appends the pre-allocated output buffer.
    shape output{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.code_object,       "code_object"),
                    f(self.symbol_name,       "symbol_name"),
                    f(self.n_groups,          "n_groups"),
                    f(self.block_size,        "block_size"),
                    f(self.pad_h,             "pad_h"),
                    f(self.pad_w,             "pad_w"),
                    f(self.has_bias,          "has_bias"),
                    f(self.activation_mode,   "activation_mode"),
                    f(self.activation_alpha,  "activation_alpha"),
                    f(self.output,            "output"));
    }

    // Non-reflected: rebuilt in finalize()
    kernel k{};

#ifdef MIGRAPHX_USE_AMDMLSS
    // Factory: obtains a conv shader binary via the AMDMLSS C API (mlssGetBinaries).
    static mlss_conv_op make_gfx12_fp32_f2x3_stride1(
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
        shape::type_t dtype);
#endif

    std::string name() const { return "gpu::mlss_conv"; }
    shape compute_shape(std::vector<shape> inputs) const;
    argument compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    void finalize(context&, const shape&, const std::vector<shape>&);
    // Pre-lowering:  [0]=input, [1]=weight              (no bias)
    //                [0]=input, [1]=weight, [2]=bias    (has_bias)
    // Post-lowering: output buffer appended as last arg
    std::vector<std::size_t> output_alias(const std::vector<shape>& inputs) const
    {
        std::size_t expected = has_bias ? 3 : 2;
        if(inputs.size() > expected)
            return {inputs.size() - 1};
        return {};
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
