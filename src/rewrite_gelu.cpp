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

/**
 * The replacement approximation is equivalent to:
 * GELU(x) ~= 0.5 * x * ( 1 + tanh( sqrt(2/M_PI) * (x + 0.044715 * x^3)))
 * You can rearrange to the form used in this by recognizing that
 * 1 + tanh(x) = (2) / (1 + exp(-2 * x)).
 * The fitting constant 0.044715 is from
 * A. Choudhury, ‘A simple approximation to the area under standard normal curve’, Mathematics and
 * Statistics, vol. 2, no. 3, pp. 147–149, 2014.
 */
void replace_with_tanh_exp_gelu(module& m, const match::matcher_result& r)
{
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

/**
 * Finds erfGELU blocks using the Gaussian distribution and replaces them with the tanh_exp
 * approximation if the data type is fp16. TODO consider also for fp8 datatype.
 */
struct find_gelu_erf
{
    auto matcher() const { return match::any_of(match::gelu_erf(), match::gelu_tanh()); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto x   = r.instructions["x"];
        auto input_type                              = x->get_shape().type();
        std::set<decltype(input_type)> convert_types = {migraphx::shape::half_type};
        if(not contains(convert_types, input_type))
            return;

        replace_with_tanh_exp_gelu(m, r);
    }
};

/**
 * Find tanhGELU blocks and replace them with a rearranged version that is less likely to overflow
 * and is more performant.
 */
struct find_tanh_fast_gelu
{
    auto matcher() const { return match::gelu_tanh(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        replace_with_tanh_exp_gelu(m, r);
    }
};

void rewrite_gelu::apply(module& m) const
{
    if(fast_math)
    {
        match::find_matches(m, find_gelu_erf{});
    }
    else
    {
        match::find_matches(m, find_tanh_fast_gelu{});
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
