/* The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct celu : op_builder<celu>
{
    float alpha = 1.0f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.alpha, "alpha"));
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        if(float_equal(alpha, 0.0f))
        {
            MIGRAPHX_THROW("celu op_builder: alpha is zero (division by zero)");
        }

        auto input_lens = args[0]->get_shape().lens();
        auto input_type = args[0]->get_shape().type();
        if(input_type != migraphx::shape::float_type)
        {
            MIGRAPHX_THROW("celu op_builder: input tensor not float type");
        }

        auto zero_lit = m.add_literal({input_type, {0.}});
        zero_lit      = m.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), zero_lit);

        auto one_lit = m.add_literal({input_type, {1.}});
        one_lit = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}),
                                    one_lit);

        auto alpha_lit = m.add_literal({input_type, {alpha}});
        alpha_lit      = m.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", input_lens}}), alpha_lit);

        auto linear_part = insert_common_op(m, ins, "max", zero_lit, args[0]);
        auto divi        = insert_common_op(m, ins, "div", args[0], alpha_lit);
        auto expo        = insert_common_op(m, ins, "exp", divi);
        auto sub         = insert_common_op(m, ins, "sub", expo, one_lit);
        auto mul         = insert_common_op(m, ins, "mul", alpha_lit, sub);
        auto exp_part    = insert_common_op(m, ins, "min", zero_lit, mul);
        return {insert_common_op(m, ins, "add", linear_part, exp_part)};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
