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

// hardsigmoid has no native op: clip(alpha * x + beta, 0, 1).
struct hardsigmoid : op_builder<hardsigmoid>
{
    float alpha = 1.0f / 6.0f;
    float beta  = 0.5f;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.alpha, "alpha"), f(self.beta, "beta"));
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x    = args[0];
        auto type = x->get_shape().type();

        auto alpha_lit = m.add_literal({type, {alpha}});
        auto beta_lit  = m.add_literal({type, {beta}});
        auto lo        = m.add_literal({type, {0.0f}});
        auto hi        = m.add_literal({type, {1.0f}});

        auto scaled  = insert_common_op(m, ins, "mul", alpha_lit, x);
        auto shifted = insert_common_op(m, ins, "add", beta_lit, scaled);
        return {insert_common_op(m, ins, "clip", shifted, lo, hi)};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
