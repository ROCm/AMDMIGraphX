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
#include <migraphx/make_op.hpp>

#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

// Split a constant tensor C into C = C_hi + C_lo where C_hi = fp16-rounded C and
// C_lo = fp16-rounded residual, returning concat(C_hi, C_lo) along `axis` as fp16.
// All folds at compile time since C is constant. Pairing this against the other
// operand duplicated along its matching contraction axis recovers, in the fp16
// accumulation, the mantissa bits that a plain fp16 cast of C would have dropped.
instruction_ref split_fp16(module& m, instruction_ref pos, instruction_ref c, std::size_t axis)
{
    auto c_hi_h =
        m.insert_instruction(pos, make_op("convert", {{"target_type", shape::half_type}}), c);
    auto c_hi_f =
        m.insert_instruction(pos, make_op("convert", {{"target_type", shape::float_type}}), c_hi_h);
    auto c_lo_f = m.insert_instruction(pos, make_op("sub"), c, c_hi_f);
    auto c_lo_h =
        m.insert_instruction(pos, make_op("convert", {{"target_type", shape::half_type}}), c_lo_f);
    return m.insert_instruction(
        pos, make_op("concat", {{"axis", axis}}), c_hi_h, c_lo_h);
}

// Cast `x` to fp16 and duplicate it along `axis` without copying: insert a
// size-1 axis, broadcast it to 2, then reshape to merge back into `axis`. Same
// layout as concat(x, x) along `axis`, so contraction over the doubled axis
// pairs the first copy with the C_hi half and the second with the C_lo half of
// split_fp16.
instruction_ref duplicate_axis(module& m, instruction_ref pos, instruction_ref x, std::size_t axis)
{
    auto x_h =
        m.insert_instruction(pos, make_op("convert", {{"target_type", shape::half_type}}), x);
    const auto& lens = x_h->get_shape().lens();
    assert(axis < lens.size());
    std::vector<std::size_t> bc_lens(lens.size() + 1);
    std::copy(lens.begin(), lens.begin() + axis, bc_lens.begin());
    bc_lens[axis] = 2;
    std::copy(lens.begin() + axis, lens.end(), bc_lens.begin() + axis + 1);
    std::vector<std::int64_t> reshape_dims(lens.begin(), lens.end());
    reshape_dims[axis] *= 2;

    auto x_unsq = m.insert_instruction(
        pos, make_op("unsqueeze", {{"axes", {axis}}}), x_h);
    auto x_bc =
        m.insert_instruction(pos, make_op("multibroadcast", {{"out_lens", bc_lens}}), x_unsq);
    return m.insert_instruction(pos, make_op("reshape", {{"dims", reshape_dims}}), x_bc);
}

void process_convolution(module& m, instruction_ref ins, std::size_t skip_small_k)
{
    const auto out_type = ins->get_shape().type();
    if(out_type != shape::float_type)
        return;

    if(ins->get_shape().dynamic())
        return;

    auto inputs = ins->inputs();
    auto x      = inputs[0];
    auto w      = inputs[1];
    if(not w->can_eval())
        return;

    // The hi/lo split below assumes a single input-channel group.
    auto op_val = ins->get_operator().to_value();
    if(op_val.contains("group") and op_val.at("group").to<int>() != 1)
        return;

    const auto& w_shape = w->get_shape();

    // Skip when conv is too small to benefit from fp16. These also tend
    // to be precision-sensitive (often follow upstream reductions whose
    // small magnitudes mean fp16 input rounding dominates absolute error).
    std::size_t reduction = std::accumulate(
        w_shape.lens().begin() + 1, w_shape.lens().end(), std::size_t{1}, std::multiplies<>());
    if(reduction < skip_small_k)
        return;

    // Split the constant weights and duplicate the input along the input-channel
    // axis (axis 1), the convolution's contraction axis.
    auto w_concat  = split_fp16(m, ins, w, 1);
    auto x_doubled = duplicate_axis(m, ins, x, 1);

    auto half_conv = m.insert_instruction(ins, ins->get_operator(), x_doubled, w_concat);
    auto converted =
        m.insert_instruction(ins, make_op("convert", {{"target_type", out_type}}), half_conv);

    m.replace_instruction(ins, converted);
}

void process_dot(module& m, instruction_ref ins, std::size_t skip_small_k)
{
    const auto out_type = ins->get_shape().type();
    if(out_type != shape::float_type)
        return;

    if(ins->get_shape().dynamic())
        return;

    auto inputs = ins->inputs();
    auto a      = inputs[0];
    auto b      = inputs[1];
    if(not a->can_eval() and not b->can_eval())
        return;

    // dot enforces same_ndims, so A and B share a rank. The contraction dim K is
    // the last axis of A and the second-to-last axis of B.
    const std::size_t rank = a->get_shape().lens().size();
    const std::size_t k    = a->get_shape().lens().back();
    if(k < skip_small_k)
        return;

    // Split whichever operand is constant and duplicate the other along its
    // matching contraction axis so the fp16 contraction computes A*B_hi + A*B_lo
    // (or A_hi*B + A_lo*B), recovering the constant's dropped mantissa bits.
    instruction_ref new_a;
    instruction_ref new_b;
    if(b->can_eval())
    {
        new_b = split_fp16(m, ins, b, rank - 2);
        new_a = duplicate_axis(m, ins, a, rank - 1);
    }
    else
    {
        new_a = split_fp16(m, ins, a, rank - 1);
        new_b = duplicate_axis(m, ins, b, rank - 2);
    }

    auto half_dot = m.insert_instruction(ins, ins->get_operator(), new_a, new_b);
    auto converted =
        m.insert_instruction(ins, make_op("convert", {{"target_type", out_type}}), half_dot);

    m.replace_instruction(ins, converted);
}

} // namespace

void fast_mm::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() == "convolution")
            process_convolution(m, ins, skip_small_k);
        else if(ins->name() == "dot")
            process_dot(m, ins, skip_small_k);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
