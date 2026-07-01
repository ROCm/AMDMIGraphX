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
#ifndef MIGRAPHX_GUARD_RTGLIB_FAST_MM_HPP
#define MIGRAPHX_GUARD_RTGLIB_FAST_MM_HPP

#include <string>
#include <migraphx/config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct module;

struct MIGRAPHX_EXPORT fast_mm
{
    std::size_t skip_small_k = 64;
    // A weight-based heuristic decides whether the cheap 2-product scheme (split only the
    // constant operand, leave the activation at plain fp16) is accurate enough for each
    // op: the estimated 2-product error is ~ fp16_unit_roundoff * input_bound *
    // max_row||w||_2 * sqrt(2 ln n_outputs) (the last factor accounts for allclose taking
    // the worst over every output element). When it stays under error_threshold the op
    // uses 2-product. When it exceeds the threshold the op is precision-sensitive, and
    // three_product selects what to do with it: if set, use the 3-product scheme (also
    // split the activation, ~2^-22 residual, at 3x vs 2x the contraction); if not, skip
    // the rewrite and leave the op in fp32.
    bool three_product = false;
    // Assumed bound on the magnitude of the (unknown) activation operand.
    double input_bound = 1.0;
    // Estimated 2-product error above which an op is treated as precision-sensitive.
    // Calibrated over a sweep of fp32 mlir configs: catches all observed 2-product
    // failures, at the cost of using the costlier scheme on a few configs that 2-product
    // would have handled (the weight/output-size estimate cannot perfectly predict the
    // input-dependent per-element cancellation that allclose is sensitive to).
    double error_threshold = 6e-3;
    std::string name() const { return "fast_mm"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
