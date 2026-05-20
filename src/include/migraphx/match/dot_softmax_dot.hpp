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
#ifndef MIGRAPHX_GUARD_MATCH_DOT_SOFTMAX_DOT_HPP
#define MIGRAPHX_GUARD_MATCH_DOT_SOFTMAX_DOT_HPP

#include <migraphx/config.hpp>
#include <migraphx/matcher.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace match {

/// Match the (undecomposed) `dot -> softmax -> dot` attention pattern, with
/// optional `mul` (scale), `add` (bias), or `where` (mask) ops between the
/// first dot and the softmax. This is the form before `rewrite_reduce`
/// decomposes softmax into its `div(exp(sub(x, max)), sum(exp(...)))` chain.
///
/// `gemm_pred` is applied to both dot operations; pass `match::any()` to
/// match any dot. `bias_pred` is applied to the optional `add` (bias) op.
///
/// Bound names: "gemm1", "gemm2", "softmax", and (when the corresponding op
/// is present) "scale", "bias", "select_const", "select_cond".
template <class GemmPred, class BiasPred>
inline auto dot_softmax_dot(GemmPred gemm_pred, BiasPred bias_pred)
{
    auto gemm1 =
        match::skip(match::name("contiguous"))(match::name("dot")(gemm_pred.bind("gemm1")));
    auto mul = match::name("mul")(
        match::nargs(2), match::either_arg(0, 1)(match::is_constant().bind("scale"), gemm1));
    auto where = match::name("where")(match::arg(2)(match::is_constant().bind("select_const")),
                                      match::arg(1)(mul),
                                      match::arg(0)(match::any().bind("select_cond")));
    auto add   = match::name("add")(
        bias_pred, match::nargs(2), match::either_arg(0, 1)(match::none_of(mul).bind("bias"), mul));
    auto softmax = match::name("softmax")(match::arg(0)(match::any_of(mul, add, gemm1, where)))
                       .bind("softmax");
    return match::name("dot")(gemm_pred.bind("gemm2"))(match::arg(0)(softmax));
}

inline auto dot_softmax_dot() { return dot_softmax_dot(match::any(), match::any()); }

} // namespace match
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
