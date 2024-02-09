/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/rewrite_gelu.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/match/gelu_erf.hpp>
#include <migraphx/match/gelu_tanh.hpp>
#include <migraphx/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_GELU_MODE)

/**
 * Finds GELU blocks using the Gaussian distribution and replaces them with the sigmoid
 * approximation if the data type is fp16.
 */
struct find_gelu_erf
{
    auto matcher() const { return match::gelu_erf(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;
        auto x   = r.instructions["x"];
        auto gelu_mode = string_value_of(MIGRAPHX_GELU_MODE{}, "tanh_exp");

        if(gelu_mode == "erf")
        {
            return;
        }
        else if(gelu_mode == "tanh_pow")
        {
            double sqrt_2_rpi = sqrt(M_2_PI);
            auto sqrt_2_rpi_lit =
                m.add_literal(literal{shape{x->get_shape().type()}, {sqrt_2_rpi}});
            auto fit_const = m.add_literal(literal{shape{x->get_shape().type()}, {0.044715f}});
            auto one       = m.add_literal(literal{shape{x->get_shape().type()}, {1.0f}});
            auto half      = m.add_literal(literal{shape{x->get_shape().type()}, {0.5f}});
            auto three     = m.add_literal(literal{shape{x->get_shape().type()}, {3.0f}});
            auto x3        = insert_common_op(m, ins, make_op("pow"), {x, three});
            auto a         = insert_common_op(m, ins, make_op("mul"), {fit_const, x3});
            auto b         = m.insert_instruction(ins, make_op("add"), x, a);
            auto c         = insert_common_op(m, ins, make_op("mul"), {sqrt_2_rpi_lit, b});
            auto tanh_ins  = m.insert_instruction(ins, make_op("tanh"), c);
            auto d         = insert_common_op(m, ins, make_op("add"), {one, tanh_ins});
            auto e         = insert_common_op(m, ins, make_op("mul"), {half, x});
            auto y         = m.insert_instruction(ins, make_op("mul"), {e, d});
            m.replace_instruction(ins, y);
        }
        else if(gelu_mode == "sig")
        {
            auto lit = m.add_literal(literal{shape{x->get_shape().type()}, {1.702f}});
            auto mul = insert_common_op(m, ins, make_op("mul"), {x, lit});
            auto sig = m.insert_instruction(ins, make_op("neg"), mul);
            sig      = m.insert_instruction(ins, make_op("exp"), sig);
            auto one = m.add_literal(literal{shape{x->get_shape().type()}, {1.0f}});
            sig      = insert_common_op(m, ins, make_op("add"), {sig, one});
            sig      = m.insert_instruction(ins, make_op("div"), x, sig);
            m.replace_instruction(ins, sig);
        }
        else
        { // alternative tanh approximation
            double sqrt_2_rpi = sqrt(M_2_PI);
            auto sqrt_2_rpi_lit =
                m.add_literal(literal{shape{x->get_shape().type()}, {sqrt_2_rpi}});
            auto fit_const = m.add_literal(literal{shape{x->get_shape().type()}, {0.044715f}});
            auto one       = m.add_literal(literal{shape{x->get_shape().type()}, {1.0f}});
            auto xb        = insert_common_op(m, ins, make_op("mul"), {x, sqrt_2_rpi_lit});
            auto a         = insert_common_op(m, ins, make_op("mul"), {xb, fit_const});
            auto b         = m.insert_instruction(ins, make_op("mul"), a, x);
            auto c         = m.insert_instruction(ins, make_op("mul"), b, x);
            auto u         = m.insert_instruction(ins, make_op("add"), c, xb);
            auto neg_u     = m.insert_instruction(ins, make_op("neg"), u);
            auto d         = m.insert_instruction(ins, make_op("sub"), neg_u, u);
            auto emu       = m.insert_instruction(ins, make_op("exp"), d);
            auto e         = insert_common_op(m, ins, make_op("add"), {one, emu});
            auto cdf       = insert_common_op(m, ins, make_op("div"), {one, e});
            auto y         = m.insert_instruction(ins, make_op("mul"), x, cdf);
            m.replace_instruction(ins, y);
        }
    }
};

/**
 * Find fastGELU blocks (where the graph already does a GELU approximation) and replace them
 * with an alternative approximation that is less likely to overflow.
 * The replacement approximation is equivalent to:
 * GELU(x) ~= 0.5 * x * ( 1 + tanh( sqrt(2/M_PI) * (x + 0.044715 * x^3)))
 * You can rearrange to the form used in this by recognizing that
 * 1 + tanh(x) = (2) / (1 + exp(-2 * x)).
 * The fitting constant 0.044715 is from
 * A. Choudhury, ‘A simple approximation to the area under standard normal curve’, Mathematics and
 * Statistics, vol. 2, no. 3, pp. 147–149, 2014.
 */
struct find_tanh_fast_gelu
{
    auto matcher() const { return match::gelu_tanh(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto gelu_mode = string_value_of(MIGRAPHX_GELU_MODE{}, "tanh_exp");
        if(gelu_mode == "tanh_pow")
        {
            return;
        }
        auto ins            = r.result;
        auto x              = r.instructions["x"];
        double sqrt_2_rpi   = sqrt(M_2_PI);
        auto sqrt_2_rpi_lit = m.add_literal(literal{shape{x->get_shape().type()}, {sqrt_2_rpi}});
        auto fit_const      = m.add_literal(literal{shape{x->get_shape().type()}, {0.044715f}});
        auto one            = m.add_literal(literal{shape{x->get_shape().type()}, {1.0f}});
        auto xb             = insert_common_op(m, ins, make_op("mul"), {x, sqrt_2_rpi_lit});
        auto a              = insert_common_op(m, ins, make_op("mul"), {xb, fit_const});
        auto b              = m.insert_instruction(ins, make_op("mul"), a, x);
        auto c              = m.insert_instruction(ins, make_op("mul"), b, x);
        auto u              = m.insert_instruction(ins, make_op("add"), c, xb);
        auto neg_u          = m.insert_instruction(ins, make_op("neg"), u);
        auto d              = m.insert_instruction(ins, make_op("sub"), neg_u, u);
        auto emu            = m.insert_instruction(ins, make_op("exp"), d);
        auto e              = insert_common_op(m, ins, make_op("add"), {one, emu});
        auto cdf            = insert_common_op(m, ins, make_op("div"), {one, e});
        auto y              = m.insert_instruction(ins, make_op("mul"), x, cdf);
        m.replace_instruction(ins, y);
    }
};

void rewrite_gelu::apply(module& m) const
{
    if(fast_math)
    {
        match::find_matches(m, find_gelu_erf{}, find_tanh_fast_gelu{});
    }
    else
    {
        match::find_matches(m, find_tanh_fast_gelu{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
