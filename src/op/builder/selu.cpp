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

#include <migraphx/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

// selu has no native op: gamma * (max(0, x) + min(0, alpha * (exp(x) - 1))).
struct selu : op_builder<selu>
{
    float alpha = 1.6732632423543772f;
    float gamma = 1.0507009873554805f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.alpha, "alpha"), f(self.gamma, "gamma"));
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x    = args[0];
        auto type = x->get_shape().type();

        auto zero      = m.add_literal({type, {0.0f}});
        auto one       = m.add_literal({type, {1.0f}});
        auto alpha_lit = m.add_literal({type, {alpha}});
        auto gamma_lit = m.add_literal({type, {gamma}});

        auto linear   = insert_common_op(m, ins, "max", zero, x);
        auto exp_x    = m.insert_instruction(ins, make_op("exp"), x);
        auto exp_sub  = insert_common_op(m, ins, "sub", exp_x, one);
        auto exp_mul  = insert_common_op(m, ins, "mul", alpha_lit, exp_sub);
        auto exp_part = insert_common_op(m, ins, "min", zero, exp_mul);
        auto sum      = insert_common_op(m, ins, "add", linear, exp_part);
        return {insert_common_op(m, ins, "mul", gamma_lit, sum)};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
