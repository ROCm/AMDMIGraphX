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
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

// Round `x` to fp16 (x_hi) and back to fp32 (x_hi_f), returning both. A barrier is
// inserted between the two converts so the lossy fp32 -> fp16 -> fp32 round-trip is not
// rewritten back to x (which would zero out the residual the split below relies on): it
// blocks eliminate_convert's find_nested_convert and precision propagation, and it
// survives fusion into the pointwise kernel. The barrier folds away with the rest of a
// constant operand, and is removed by eliminate_barrier before code generation for the
// non-constant one.
std::pair<instruction_ref, instruction_ref>
fp16_round_trip(module& m, instruction_ref pos, instruction_ref x)
{
    auto x_hi =
        m.insert_instruction(pos, make_op("convert", {{"target_type", shape::half_type}}), x);
    auto x_hi_barrier = m.insert_instruction(pos, make_op("barrier", {{"tag", "fast_mm"}}), x_hi);
    auto x_hi_f       = m.insert_instruction(
        pos, make_op("convert", {{"target_type", shape::float_type}}), x_hi_barrier);
    return {x_hi, x_hi_f};
}

// Split a constant tensor C into C = C_hi + C_lo where C_hi = fp16-rounded C and
// C_lo = fp16-rounded residual, returning the fp16 halves concatenated along `axis`:
// concat(C_hi, C_lo) for the 2-product scheme, or concat(C_hi, C_lo, C_hi) for the
// 3-product scheme (the trailing C_hi pairs with the other operand's low half).
// All folds at compile time since C is constant. Pairing this against the other
// operand duplicated along its matching contraction axis recovers, in the quant
// op's contraction, the mantissa bits that a plain fp16 cast of C would have dropped.
instruction_ref
split_fp16(module& m, instruction_ref pos, instruction_ref c, std::size_t axis, bool three_product)
{
    auto [c_hi_h, c_hi_f] = fp16_round_trip(m, pos, c);
    auto c_lo_f           = m.insert_instruction(pos, make_op("sub"), c, c_hi_f);
    auto c_lo_h =
        m.insert_instruction(pos, make_op("convert", {{"target_type", shape::half_type}}), c_lo_f);
    if(three_product)
        return m.insert_instruction(
            pos, make_op("concat", {{"axis", axis}}), c_hi_h, c_lo_h, c_hi_h);
    return m.insert_instruction(pos, make_op("concat", {{"axis", axis}}), c_hi_h, c_lo_h);
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

    auto x_unsq = m.insert_instruction(pos, make_op("unsqueeze", {{"axes", {axis}}}), x_h);
    auto x_bc =
        m.insert_instruction(pos, make_op("multibroadcast", {{"out_lens", bc_lens}}), x_unsq);
    return m.insert_instruction(pos, make_op("reshape", {{"dims", reshape_dims}}), x_bc);
}

// 3-product counterpart of duplicate_axis for the non-constant operand X: split X
// into fp16 hi/lo halves (X_hi = fp16(X), X_lo = fp16(X - fp32(X_hi))) and lay them
// out along the contraction `axis` as concat(X_hi, X_hi, X_lo). Paired against a
// constant split as concat(C_hi, C_lo, C_hi), the contraction computes
// X_hi*C_hi + X_hi*C_lo + X_lo*C_hi, recovering both operands' dropped mantissa bits
// (the omitted X_lo*C_lo term is ~2^-22 of the result). Unlike split_fp16 this runs
// at eval time since X is not constant.
instruction_ref split_input(module& m, instruction_ref pos, instruction_ref x, std::size_t axis)
{
    auto [x_hi, x_hi_f] = fp16_round_trip(m, pos, x);
    auto x_lo_f         = m.insert_instruction(pos, make_op("sub"), x, x_hi_f);
    auto x_lo =
        m.insert_instruction(pos, make_op("convert", {{"target_type", shape::half_type}}), x_lo_f);
    return m.insert_instruction(pos, make_op("concat", {{"axis", axis}}), x_hi, x_hi, x_lo);
}

void process_convolution(module& m,
                         instruction_ref ins,
                         std::size_t skip_small_k,
                         bool three_product)
{
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
    // axis (axis 1), the convolution's contraction axis. The 3-product scheme also
    // splits the input into hi/lo halves to recover its dropped mantissa bits.
    auto w_concat  = split_fp16(m, ins, w, 1, three_product);
    auto x_doubled = three_product ? split_input(m, ins, x, 1) : duplicate_axis(m, ins, x, 1);

    // quant_convolution takes the fp16 operands and accumulates directly into the
    // fp32 output, so no trailing convert is needed.
    auto quant_conv = m.insert_instruction(
        ins, make_op("quant_convolution", ins->get_operator().to_value()), x_doubled, w_concat);

    m.replace_instruction(ins, quant_conv);
}

void process_dot(module& m, instruction_ref ins, std::size_t skip_small_k, bool three_product)
{
    auto inputs           = ins->inputs();
    auto a                = inputs[0];
    auto b                = inputs[1];
    const bool a_can_eval = a->can_eval();
    const bool b_can_eval = b->can_eval();
    if(not a_can_eval and not b_can_eval)
        return;

    // dot enforces same_ndims, so A and B share a rank. The contraction dim K is
    // the last axis of A and the second-to-last axis of B.
    const std::size_t rank = a->get_shape().lens().size();
    const std::size_t k    = a->get_shape().lens().back();
    if(k < skip_small_k)
        return;

    // Split whichever operand is constant and duplicate the other along its
    // matching contraction axis so the contraction computes A*B_hi + A*B_lo
    // (or A_hi*B + A_lo*B), recovering the constant's dropped mantissa bits. The
    // 3-product scheme also splits the non-constant operand to recover its bits.
    instruction_ref new_a;
    instruction_ref new_b;
    if(b_can_eval)
    {
        new_b = split_fp16(m, ins, b, rank - 2, three_product);
        new_a =
            three_product ? split_input(m, ins, a, rank - 1) : duplicate_axis(m, ins, a, rank - 1);
    }
    else
    {
        new_a = split_fp16(m, ins, a, rank - 1, three_product);
        new_b =
            three_product ? split_input(m, ins, b, rank - 2) : duplicate_axis(m, ins, b, rank - 2);
    }

    // quant_dot takes the fp16 operands and accumulates directly into the fp32
    // output, so no trailing convert is needed.
    auto q_dot = m.insert_instruction(ins, make_op("quant_dot"), new_a, new_b);

    m.replace_instruction(ins, q_dot);
}

} // namespace

void fast_mm::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->get_shape().type() != shape::float_type)
            continue;

        if(ins->get_shape().dynamic())
            continue;
        if(ins->name() == "convolution")
            process_convolution(m, ins, skip_small_k, three_product);
        else if(ins->name() == "dot")
            process_dot(m, ins, skip_small_k, three_product);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
