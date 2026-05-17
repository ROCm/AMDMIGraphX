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
#include <migraphx/fast_mm.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void fast_mm::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "convolution")
            continue;

        const auto out_type = ins->get_shape().type();
        if(out_type != shape::float_type)
            continue;

        const auto& out_shape = ins->get_shape();
        if(out_shape.dynamic())
            continue;

        auto inputs = ins->inputs();
        auto x      = inputs[0];
        auto w      = inputs[1];
        if(not w->can_eval())
            continue;

        // The hi/lo split below assumes a single input-channel group.
        auto op_val = ins->get_operator().to_value();
        if(op_val.contains("group") and op_val.at("group").to<int>() != 1)
            continue;

        const auto& w_shape = w->get_shape();

        // Skip when conv is too small to benefit from fp16. These also tend
        // to be precision-sensitive (often follow upstream reductions whose
        // small magnitudes mean fp16 input rounding dominates absolute error).
        std::size_t reduction = 1;
        for(std::size_t i = 1; i < w_shape.ndim(); ++i)
            reduction *= w_shape.lens()[i];
        if(out_shape.elements() * reduction < 1024)
            continue;

        // W = W_hi + W_lo where W_hi = fp16-rounded W and W_lo = fp16-rounded
        // residual. All folds at compile time since W is constant. Concat
        // along input-channel axis and duplicate X so the fp16 conv computes
        // X*W_hi + X*W_lo = X*(W_hi+W_lo) ≈ X*W, recovering ~fp32 precision on
        // the W side. (Experiments show the equivalent hi/lo split on X gives
        // no further gain on this kernel — the multiplication itself appears
        // to round to fp16 before accumulation, so per-multiply X rounding is
        // a hard floor that input-side tricks can't recover.)
        auto w_hi_h =
            m.insert_instruction(ins, make_op("convert", {{"target_type", shape::half_type}}), w);
        auto w_hi_f = m.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::float_type}}), w_hi_h);
        auto w_lo_f = m.insert_instruction(ins, make_op("sub"), w, w_hi_f);
        auto w_lo_h = m.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::half_type}}), w_lo_f);
        auto w_concat = m.insert_instruction(ins, make_op("concat", {{"axis", 1}}), w_hi_h, w_lo_h);

        auto x_h =
            m.insert_instruction(ins, make_op("convert", {{"target_type", shape::half_type}}), x);

        // Duplicate X along the input-channel axis without copying: insert a
        // size-1 axis, broadcast it to 2, then reshape to merge back into the
        // channel dim. Same semantics as concat(X_h, X_h) along axis 1.
        const auto& x_lens = x_h->get_shape().lens();
        std::vector<std::size_t> bc_lens(x_lens.size() + 1);
        bc_lens[0] = x_lens[0];
        bc_lens[1] = 2;
        std::copy(x_lens.begin() + 1, x_lens.end(), bc_lens.begin() + 2);
        std::vector<std::int64_t> reshape_dims(x_lens.begin(), x_lens.end());
        reshape_dims[1] *= 2;

        auto x_unsq = m.insert_instruction(ins, make_op("unsqueeze", {{"axes", {1}}}), x_h);
        auto x_bc =
            m.insert_instruction(ins, make_op("multibroadcast", {{"out_lens", bc_lens}}), x_unsq);
        auto x_doubled =
            m.insert_instruction(ins, make_op("reshape", {{"dims", reshape_dims}}), x_bc);

        auto half_conv = m.insert_instruction(ins, ins->get_operator(), x_doubled, w_concat);
        auto converted =
            m.insert_instruction(ins, make_op("convert", {{"target_type", out_type}}), half_conv);

        m.replace_instruction(ins, converted);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
