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
#ifndef MIGRAPHX_GUARD_OPERATORS_QGEMM_HPP
#define MIGRAPHX_GUARD_OPERATORS_QGEMM_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/value.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/**
 * Weight-only quantized GEMM for the DxGML weight-only quantization (WoQ) pattern.
 *
 * inputs[0] : activation  — fp16 or bf16
 * inputs[1] : weight      — int8 or uint4 (quantized, stored as-is)
 * inputs[2] : weight scale — fp16 or bf16 (scalar, per-channel, or block-quantized)
 * inputs[3] : weight zero-point — same type as inputs[1], optional
 *
 * Output type matches the activation type (fp16 or bf16).
 *
 * The dequant formula  scale * (weight - zp)  is applied INSIDE the GPU kernel;
 * the activation tensor is used as-is without any pre-dequantization.
 *
 * Lowered to rocMLIR which generates an in-kernel dequant+GEMM kernel.
 */
struct qgemm
{
    value attributes() const { return {{"general_data_type", "dot"}}; }

    std::string name() const { return "qgemm"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this}.has(3, 4);

        const shape& act    = inputs[0]; // fp16/bf16 activation
        const shape& weight = inputs[1]; // int8/uint4 weight

        if(act.lens().size() < 2)
            MIGRAPHX_THROW("QGEMM: activation must be >= 2D");
        if(weight.lens().size() < 2)
            MIGRAPHX_THROW("QGEMM: weight must be >= 2D");

        // Verify inner dimensions contract
        std::size_t act_k    = act.lens()[act.lens().size() - 1];
        std::size_t weight_k = weight.lens()[weight.lens().size() - 2];
        if(act_k != weight_k)
            MIGRAPHX_THROW("QGEMM: inner dimensions do not match: act_k=" +
                           std::to_string(act_k) + " weight_k=" + std::to_string(weight_k));

        // Output shape: batch dims from act, M from act, N from weight's last dim
        auto out_lens         = act.lens();
        out_lens.back()       = weight.lens().back();

        // Output type matches activation type
        return {act.type(), out_lens};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_OPERATORS_QGEMM_HPP
