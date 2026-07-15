/* The MIT License (MIT)
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

#include <cmath>
#include <migraphx/float_equal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// vector_norm has no native op: reduce abs(x) over axes per the ord-specific formula.
struct vector_norm : op_builder<vector_norm>
{
    float ord                 = 2.0f;
    std::vector<int64_t> axes = {};
    bool keepdim              = false;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.ord, "ord"), f(self.axes, "axes"), f(self.keepdim, "keepdim"));
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x      = args[0];
        auto x_type = x->get_shape().type();
        auto abs_x  = m.insert_instruction(ins, make_op("abs"), x);

        instruction_ref out;
        if(float_equal(ord, 0.0f))
        {
            // count of nonzero elements: sum(abs(x) > 0)
            auto zero    = m.add_literal(migraphx::literal{migraphx::shape{x_type}, {0.0f}});
            auto nonzero = insert_common_op(m, ins, "greater", abs_x, zero);
            auto counts =
                m.insert_instruction(ins, make_op("convert", {{"target_type", x_type}}), nonzero);
            out = m.insert_instruction(ins, make_op("reduce_sum", {{"axes", axes}}), counts);
        }
        else if(std::isinf(ord))
        {
            // +inf -> max(abs(x)), -inf -> min(abs(x))
            auto reduce = ord > 0 ? "reduce_max" : "reduce_min";
            out         = m.insert_instruction(ins, make_op(reduce, {{"axes", axes}}), abs_x);
        }
        else
        {
            // sum(abs(x) ^ ord) ^ (1 / ord)
            auto ord_lit = m.add_literal(migraphx::literal{migraphx::shape{x_type}, {ord}});
            auto pow_x   = insert_common_op(m, ins, "pow", abs_x, ord_lit);
            auto sum_pow =
                m.insert_instruction(ins, make_op("reduce_sum", {{"axes", axes}}), pow_x);
            auto recip = m.insert_instruction(ins, make_op("recip"), ord_lit);
            out        = insert_common_op(m, ins, "pow", sum_pow, recip);
        }

        if(not keepdim)
            out = m.insert_instruction(ins, make_op("squeeze", {{"axes", axes}}), out);
        return {out};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
