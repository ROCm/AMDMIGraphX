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

#include <migraphx/rewrite_gelu.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct find_gelu_erf
{
    static auto match_div()
    {
        return match::name("div")(match::either_arg(0, 1)(
            match::any().bind("x"), match::skip_broadcasts(match::has_value(1.414f, 1e-3))));
    }

    static auto match_erf() { return match::name("erf")(match::arg(0)(match_div())); }

    static auto match_add()
    {
        return match::name("add")(
            match::either_arg(0, 1)(match_erf(), match::skip_broadcasts(match::has_value(1.0f))));
    }

    static auto match_mul()
    {
        return match::name("mul")(match::either_arg(0, 1)(match::any(), match_add()));
    }

    auto matcher() const
    {
        return match::name("mul")(
            match::either_arg(0, 1)(match_mul(), match::skip_broadcasts(match::has_value(0.5f))));
    }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;
        auto x   = r.instructions["x"];

        auto lit = m.add_literal(literal{shape{x->get_shape().type()}, {1.702f}});
        auto mul = m.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", x->get_shape().lens()}}), lit);
        mul      = m.insert_instruction(ins, make_op("mul"), x, mul);
        auto sig = m.insert_instruction(ins, make_op("neg"), mul);
        sig      = m.insert_instruction(ins, make_op("exp"), sig);
        auto one = m.add_literal(literal{shape{x->get_shape().type()}, {1.0f}});
        one      = m.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", x->get_shape().lens()}}), one);
        sig = m.insert_instruction(ins, make_op("add"), sig, one);
        sig = m.insert_instruction(ins, make_op("div"), one, sig);
        sig = m.insert_instruction(ins, make_op("mul"), x, sig);
        m.replace_instruction(ins, sig);
    }
};

void rewrite_gelu::apply(module& m) const { match::find_matches(m, find_gelu_erf{}); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
