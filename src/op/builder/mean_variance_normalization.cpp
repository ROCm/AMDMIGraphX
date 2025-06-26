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
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {
namespace builder {

struct mean_variance_normalization : op_builder<mean_variance_normalization>
{
    std::vector<int64_t> axes{0, 2, 3};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.axes, "axes"));
    }

    std::vector<instruction_ref>
    insert(module& m, instruction_ref ins, const std::vector<instruction_ref>& args) const
    {
        auto x = args.front();
        if(axes.size() != x->get_shape().ndim() - 1)
            MIGRAPHX_THROW("mvn op_builder: Length of axes attribute needs to be equal to input "
                           "tensor rank - 1");

        auto x_mean = m.insert_instruction(ins, make_op("reduce_mean", {{"axes", axes}}), x);
        auto x_mean_squared = insert_common_op(m, ins, "mul", x_mean, x_mean);

        auto x_squared = insert_common_op(m, ins, "mul", x, x);
        auto x_squared_mean =
            m.insert_instruction(ins, make_op("reduce_mean", {{"axes", axes}}), x_squared);

        auto mean_sub = insert_common_op(m, ins, "sub", x_squared_mean, x_mean_squared);
        auto std      = insert_common_op(m, ins, "sqrt", mean_sub);

        auto dividend = insert_common_op(m, ins, "sub", x, x_mean);
        auto epsilon  = m.add_literal(
            {x->get_shape().type(), {x->get_shape().type() == shape::half_type ? 1e-7 : 1e-9}});
        auto divisor = insert_common_op(m, ins, "add", std, epsilon);

        return {insert_common_op(m, ins, "div", dividend, divisor)};
    }
};

} // namespace builder
} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
