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
#include <migraphx/fuse_dxgml_dequant.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/op/qgemm.hpp>
#include <migraphx/op/qconv.hpp>
#include <migraphx/match/dq_helpers.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace {

// Collect instructions between a dequantizelinear and the quantized op arg.
// Returns them in reverse order (closest to qop_arg first).
static std::vector<instruction_ref> get_between_ins(const instruction_ref dqins,
                                                    const instruction_ref qop_arg)
{
    auto prev_ins = qop_arg;
    std::vector<instruction_ref> ins_between;
    while(prev_ins != dqins)
    {
        ins_between.push_back(prev_ins);
        prev_ins = prev_ins->inputs().front();
    }
    return ins_between;
}

// Replay broadcast/transpose/reshape instructions between dqins and the qop arg,
// but applied to input_ins (the quantized tensor or scale) instead of the dq output.
static instruction_ref propagate_quantized_ins(module& m,
                                               const instruction_ref dqins,
                                               instruction_ref input_ins,
                                               const std::vector<instruction_ref>& ins_between)
{
    for(auto ins : reverse_iterator_for(ins_between))
    {
        input_ins = m.insert_instruction(dqins, (*ins)->get_operator(), {input_ins});
    }
    return input_ins;
}

// Return true when zero_point is all-zeros (symmetric quantization — no correction needed).
static bool is_symmetric_zero_point(instruction_ref zp)
{
    if(not zp->can_eval())
        return false;
    bool all_zeros = false;
    zp->eval().visit([&](auto z) {
        all_zeros = std::all_of(z.begin(), z.end(), [](auto v) { return float_equal(v, 0); });
    });
    return all_zeros;
}

// Core apply logic shared between dot and convolution fusions.
//
// For dot (weight-only WoQ):
//   %w_dq = dequantizelinear(%w_q, %scale [, %zp])
//   %out  = dot(%act_f16, %w_dq)
// Transformed into:
//   %out  = qgemm(%act_f16, %w_q, %scale [, %zp])
//
// For convolution (weight-only WoQ):
//   %w_dq = dequantizelinear(%w_q, %scale [, %zp])
//   %out  = convolution(%act_f16, %w_dq)
// Transformed into:
//   %out  = qconv(%act_f16, %w_q, %scale [, %zp])
//
// Both qgemm and qconv accept mixed types (fp16 act + int8/uint4 weight) and
// perform dequantization inside the rocMLIR-generated kernel.
// The dequant formula scale*(q-zp) is reproduced entirely within the fused kernel.
static void fuse_weight_only(module& m,
                             instruction_ref qop,
                             instruction_ref dq_w)
{
    auto w_q  = dq_w->inputs()[0]; // raw quantized weight (uint4/int8)
    auto w_sc = dq_w->inputs()[1]; // weight scale
    const bool has_zp = dq_w->inputs().size() == 3;
    auto w_zp = has_zp ? dq_w->inputs()[2] : instruction_ref{};

    auto act = qop->inputs()[0]; // float16/bf16 activation — left unchanged

    // Collect reshape/broadcast/transpose between dq_w output and qop arg(1)
    auto between   = get_between_ins(dq_w, qop->inputs()[1]);
    auto w_q_prop  = propagate_quantized_ins(m, dq_w, w_q,  between);
    auto w_sc_prop = propagate_quantized_ins(m, dq_w, w_sc, between);

    instruction_ref result;

    if(qop->name() == "dot")
    {
        // qgemm: {act, w_quant, w_scale [, w_zp]}
        // Dequant (scale*(q-zp)) happens inside the rocMLIR kernel — no separate correction.
        std::vector<instruction_ref> qargs = {act, w_q_prop, w_sc_prop};
        if(has_zp and not is_symmetric_zero_point(w_zp))
        {
            auto w_zp_prop = propagate_quantized_ins(m, dq_w, w_zp, between);
            qargs.push_back(w_zp_prop);
        }
        result = m.insert_instruction(qop, make_op("qgemm"), qargs);
    }
    else
    {
        // qconv: {act, w_quant, w_scale [, w_zp]}
        // Dequant (scale*(q-zp)) happens inside the rocMLIR kernel — no separate correction.
        auto cv = qop->get_operator().to_value();
        std::vector<instruction_ref> qargs = {act, w_q_prop, w_sc_prop};
        if(has_zp and not is_symmetric_zero_point(w_zp))
        {
            auto w_zp_prop = propagate_quantized_ins(m, dq_w, w_zp, between);
            qargs.push_back(w_zp_prop);
        }
        result = m.insert_instruction(qop, make_op("qconv", cv), qargs);
    }

    m.replace_instruction(qop, result);
}

// Matcher + apply for weight-only dot fusion.
struct fuse_weight_only_dot
{
    auto matcher() const
    {
        // arg(1) feeds through a dequantize; arg(0) does NOT (weight-only pattern).
        auto dq_weight = match::arg(1)(
            match::skip_post_dq_ops(match::name("dequantizelinear").bind("dq_w")));
        auto float_act = match::arg(0)(match::none_of(
            match::skip_post_dq_ops(match::name("dequantizelinear"))));
        return match::name("dot")(dq_weight, float_act);
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        fuse_weight_only(m, r.result, r.instructions["dq_w"]);
    }
};

// Matcher + apply for weight-only convolution fusion.
struct fuse_weight_only_conv
{
    auto matcher() const
    {
        auto dq_weight = match::arg(1)(
            match::skip_post_dq_ops(match::name("dequantizelinear").bind("dq_w")));
        auto float_act = match::arg(0)(match::none_of(
            match::skip_post_dq_ops(match::name("dequantizelinear"))));
        return match::name("convolution")(dq_weight, float_act);
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        fuse_weight_only(m, r.result, r.instructions["dq_w"]);
    }
};

} // namespace

void fuse_dxgml_dequant::apply(module& m) const
{
    match::find_matches(m, fuse_weight_only_dot{});
    match::find_matches(m, fuse_weight_only_conv{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
