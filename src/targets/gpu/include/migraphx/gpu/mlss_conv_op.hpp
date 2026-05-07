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

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.code_object, "code_object"),
                    f(self.symbol_name, "symbol_name"),
                    f(self.n_groups,    "n_groups"),
                    f(self.block_size,  "block_size"),
                    f(self.pad_h,       "pad_h"),
                    f(self.pad_w,       "pad_w"));
    }

    // Non-reflected: rebuilt in finalize()
    kernel k{};

    // Factory: constructs an op pre-loaded with the GFX12 fp32 conv shader.
    static mlss_conv_op make_gfx12_fp32_f2x3_stride1();
    // Factory: constructs an op pre-loaded with the GFX12 fp32 f3x2 ostride2 conv shader.
    static mlss_conv_op make_gfx12_fp32_f3x2_ostride2();
    // Factory: constructs an op pre-loaded with the NAVI48 fp16pk conv shader.
    static mlss_conv_op make_navi48_fp16pk_f2x3_stride1();

    std::string name() const { return "gpu::mlss_conv"; }
    shape compute_shape(std::vector<shape> inputs) const;
    argument compute(context& ctx, const shape& output_shape, const std::vector<argument>& args) const;
    void finalize(context&, const shape&, const std::vector<shape>&);
    // args layout: [0]=input, [1]=weight, [2]=output
    std::vector<std::size_t> output_alias(const std::vector<shape>&) const
    {
        return {2};
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
