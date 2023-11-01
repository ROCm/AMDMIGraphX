/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/simplify_qdq.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/op/convolution.hpp>
#include <migraphx/op/quant_convolution.hpp>
#include <migraphx/op/dot.hpp>
#include <migraphx/op/quant_dot.hpp>
#include <migraphx/register_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

std::unordered_set<std::string> get_quantizable_op_names()
{
    static std::unordered_set<std::string> s = {"convolution", "dot"};
    return s;
}

struct match_find_quantizable_ops
{
    static bool
    is_valid_scale(instruction_ref scale, std::vector<std::size_t> lens, std::size_t axis)
    {
        return scale->get_shape().scalar() or scale->get_shape().elements() == lens.at(axis);
    }

    static bool is_valid_zero_point(instruction_ref zp)
    {
        if(not zp->can_eval())
            return false;

        bool all_zeros = false;
        zp->eval().visit([&](auto z) {
            all_zeros =
                std::all_of(z.begin(), z.end(), [&](auto val) { return float_equal(val, 0); });
        });
        return all_zeros;
    }

    static auto
    scale_broadcast_op(instruction_ref scale, std::vector<std::size_t> lens, std::size_t axis)
    {
        if(scale->get_shape().scalar())
        {
            return migraphx::make_op("multibroadcast", {{"out_lens", lens}});
        }
        else
        {
            return migraphx::make_op("broadcast", {{"out_lens", lens}, {"axis", axis}});
        }
    }

    // Helper function to insert quantized versions of any broadcasts and transpose ops that
    // occur between dequantizelinear and the quantized op
    static auto
    propagate_quantized_ins(module& m, const instruction_ref dqins, const instruction_ref qop)
    {
        auto qinp     = dqins->inputs().front();
        auto next_ins = dqins;

        while(next_ins != qop)
        {
            if(next_ins->name() != "dequantizelinear")
            {
                qinp = m.insert_instruction(qop, next_ins->get_operator(), qinp);
            }
            next_ins = next_ins->outputs().front();
        }
        return qinp;
    }

    static auto dequantizelinear_op(const std::string& scale, const std::string& zp)
    {
        return match::name("dequantizelinear")(
            match::arg(0)(match::skip(match::name("quantizelinear"))(match::any())),
            match::arg(1)(match::skip_broadcasts(match::is_constant().bind(scale))),
            match::arg(2)(match::skip_broadcasts(match::is_constant().bind(zp))));
    }

    auto matcher() const
    {
        return match::name(get_quantizable_op_names())(
            match::arg(0)(match::skip_broadcasts_transposes(
                dequantizelinear_op("scale1", "zp1").bind("dq1"))),
            match::arg(1)(match::skip_broadcasts_transposes(
                dequantizelinear_op("scale2", "zp2").bind("dq2"))));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto qop    = r.result;
        auto dq1    = r.instructions["dq1"];
        auto dq2    = r.instructions["dq2"];
        auto scale1 = r.instructions["scale1"];
        auto scale2 = r.instructions["scale2"];
        auto zp1    = r.instructions["zp1"];
        auto zp2    = r.instructions["zp2"];

        // Only INT8 type currently supported
        if(dq1->inputs().front()->get_shape().type() != migraphx::shape::int8_type or
           dq2->inputs().front()->get_shape().type() != migraphx::shape::int8_type)
            return;

        // Only symmetric quantization supported (ie. non-zero zero_points not allowed)
        if(not(is_valid_zero_point(zp1) and is_valid_zero_point(zp2)))
            return;

        // Only support scalar and 1D scales
        if(scale1->get_shape().lens().size() != 1 or scale2->get_shape().lens().size() != 1)
            return;

        // Propagate q1 and q2 through any broadcasts and transposes before qop
        auto qop_args  = qop->inputs();
        qop_args.at(0) = propagate_quantized_ins(m, dq1, qop);
        qop_args.at(1) = propagate_quantized_ins(m, dq2, qop);
        instruction_ref dq;
        instruction_ref out_scale;
        instruction_ref zero_point;
        if(qop->name() == "convolution")
        {
            auto conv_val = qop->get_operator().to_value();
            dq            = m.insert_instruction(
                qop, migraphx::make_op("quant_convolution", conv_val), qop_args);
            auto out_lens = dq->get_shape().lens();

            // Input scale should always be scalar and weight scale can be scalar or 1D of the
            // same lens as the output channel dim (dim 1 in the output)
            if(not(is_valid_scale(scale1, out_lens, 1) and is_valid_scale(scale2, out_lens, 1)))
                return;

            auto s1_bcast =
                m.insert_instruction(qop, scale_broadcast_op(scale1, out_lens, 1), scale1);
            auto s2_bcast =
                m.insert_instruction(qop, scale_broadcast_op(scale2, out_lens, 1), scale2);

            out_scale = m.insert_instruction(qop, migraphx::make_op("mul"), s1_bcast, s2_bcast);
        }
        else if(qop->name() == "dot")
        {
            dq            = m.insert_instruction(qop, migraphx::make_op("quant_dot"), qop_args);
            auto out_lens = dq->get_shape().lens();

            // For (..., M, N) x (..., N, K) dot, only support cases where quantization axis is M
            // for input1 and K for input 2
            if(not(is_valid_scale(scale1, out_lens, out_lens.size() - 2) and
                   is_valid_scale(scale2, out_lens, out_lens.size() - 1)))
                return;

            auto s1_bcast = m.insert_instruction(
                qop, scale_broadcast_op(scale1, out_lens, out_lens.size() - 2), scale1);
            auto s2_bcast = m.insert_instruction(
                qop, scale_broadcast_op(scale2, out_lens, out_lens.size() - 1), scale2);

            out_scale = m.insert_instruction(qop, migraphx::make_op("mul"), s1_bcast, s2_bcast);
        }

        dq = m.insert_instruction(qop, make_op("dequantizelinear"), dq, out_scale);
        m.replace_instruction(qop, dq);
    }
};

bool compare_literals(instruction_ref ins1, instruction_ref ins2)
{
    if(ins1->name() == "broadcast" or ins1->name() == "multibroadcast")
        ins1 = ins1->inputs().front();
    auto x = ins1->eval();
    if(x.empty())
        return false;
    auto literal1 = ins1->get_literal();
    if(ins2->name() == "broadcast" or ins2->name() == "multibroadcast")
        ins2 = ins2->inputs().front();
    auto y = ins2->eval();
    if(y.empty())
        return false;
    auto literal2 = ins2->get_literal();

    bool diff_shapes_equal_vals = false;
    visit_all(ins1->get_literal(), ins2->get_literal())([&](const auto l1, const auto l2) {
        diff_shapes_equal_vals =
            std::all_of(
                l1.begin() + 1, l1.end(), [&](auto v) { return float_equal(v, l1.front()); }) and
            std::all_of(l2.begin(), l2.end(), [&](auto v) { return float_equal(v, l1.front()); });
    });

    return (x == y) or diff_shapes_equal_vals;
}

void remove_qdq_pairs(module& m)
{
    for(auto ins : iterator_for(m))
    {
        auto args = ins->inputs();
        for(auto&& arg : args)
        {
            if(arg->name() == "dequantizelinear")
            {
                auto q = arg->inputs().front();
                if((q->name() == "quantizelinear") and
                   compare_literals(arg->inputs().at(1), q->inputs().at(1)) and
                   compare_literals(arg->inputs().at(2), q->inputs().at(2)))
                {
                    instruction::replace_argument(ins, arg, q->inputs().front());
                }
            }
        }
    }
}

void simplify_qdq::apply(module& m) const
{
    match::find_matches(m, match_find_quantizable_ops{});
    migraphx::run_passes(m, {migraphx::dead_code_elimination{}});
    remove_qdq_pairs(m);
    migraphx::run_passes(m, {migraphx::dead_code_elimination{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
