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
#ifndef MIGRAPHX_GUARD_FUSE_DXGML_DEQUANT_HPP
#define MIGRAPHX_GUARD_FUSE_DXGML_DEQUANT_HPP

#include <migraphx/config.hpp>
#include <migraphx/module.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * DxGML-specific pass: fuse weight-only dequantizelinear into dot/convolution.
 *
 * Applied immediately after DxGML MLIR parsing (not in the general GPU pipeline).
 *
 * Pattern matched:
 *   %w_dq = dequantizelinear(%w_quant, %scale [, %zp])   // weight-only
 *   %out  = dot(%activation_f16, %w_dq)                  // activation NOT dequantized
 *
 * Transformed to a 4-input quant_dot:
 *   %out  = quant_dot(%activation_f16, %w_quant, %unit_scale=1, %w_scale)
 * followed by a zero-point correction if zp != 0:
 *   %out  = %out - quant_dot(%activation_f16, %w_zp, 1, %w_scale)
 * and a type convert if the original output type differs from float32.
 *
 * The dequant formula `scale * (q - zp)` is faithfully reproduced in the fused form.
 * For block-quantized weights the scale tensor (e.g. {96,5120} for a {3072,5120} weight)
 * is passed through as-is; the quant kernel handles the broadcast internally.
 *
 * The same fusion applies to convolution via quant_convolution.
 */
struct MIGRAPHX_EXPORT fuse_dxgml_dequant
{
    std::string name() const { return "fuse_dxgml_dequant"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_FUSE_DXGML_DEQUANT_HPP
