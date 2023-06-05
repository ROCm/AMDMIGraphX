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
#include <migraphx/match/gelu_erf.hpp>
#include <migraphx/common.hpp>

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_USE_FAST_GELU)

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct find_gelu_erf
{
    auto matcher() const { return match::gelu_erf(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto ins = r.result;
        auto x   = r.instructions["x"];
        if(x->get_shape().type() != migraphx::shape::half_type)
            return;
        if(enabled(MIGRAPHX_USE_FAST_GELU{})) {
            auto lit = m.add_literal(literal{shape{x->get_shape().type()}, {1.702f}});
            auto mul = insert_common_op(m, ins, make_op("mul"), {x, lit});
            auto sig = m.insert_instruction(ins, make_op("neg"), mul);
            sig      = m.insert_instruction(ins, make_op("exp"), sig);
            auto one = m.add_literal(literal{shape{x->get_shape().type()}, {1.0f}});
            sig      = insert_common_op(m, ins, make_op("add"), {sig, one});
            sig      = m.insert_instruction(ins, make_op("div"), x, sig);
            m.replace_instruction(ins, sig);
        } else {
            auto lit1 = m.add_literal(literal{shape{x->get_shape().type()}, {0.7978845608028654f}});// sqrt(2.0/M_PI)
            auto lit2 = m.add_literal(literal{shape{x->get_shape().type()}, {0.035677408136300125f}}); // 0.044715 * sqrt(2.0/M_PI)
            auto one = m.add_literal(literal{shape{x->get_shape().type()}, {1.0f}});
            auto two = m.add_literal(literal{shape{x->get_shape().type()}, {2.0f}});
            auto three = m.add_literal(literal{shape{x->get_shape().type()}, {3.0f}});
            auto mul1 = insert_common_op(m, ins, make_op("mul"), {x, lit1}); 
            auto x_3 = insert_common_op(m, ins, make_op("pow"), {x, three}); // x^3
            auto mul2 = insert_common_op(m, ins, make_op("mul"), {x_3, lit2});
            auto add1 = insert_common_op(m, ins, make_op("add"), {mul1, mul2}); 
            auto tanh_ins = insert_common_op(m, ins, make_op("tanh"), {add1});
            auto add2 = insert_common_op(m, ins, make_op("add"), {one, tanh_ins});
            auto x_div = insert_common_op(m, ins, make_op("div"), {x, two});
            auto gelu_tanh = insert_common_op(m, ins, make_op("mul"), {x_div, add2});
            m.replace_instruction(ins, gelu_tanh);
        }
    }
};

void rewrite_gelu::apply(module& m) const { match::find_matches(m, find_gelu_erf{}); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
