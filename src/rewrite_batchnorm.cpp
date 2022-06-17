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
#include <migraphx/rewrite_batchnorm.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/batch_norm_inference.hpp>
#include <migraphx/op/broadcast.hpp>
#include <migraphx/op/add.hpp>
#include <migraphx/op/mul.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/dfor.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void rewrite_batchnorm::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "batch_norm_inference")
            continue;
        // Get scale, bias, mean, variance from inputs
        auto gamma    = ins->inputs()[1]->eval();
        auto bias     = ins->inputs()[2]->eval();
        auto mean     = ins->inputs()[3]->eval();
        auto variance = ins->inputs()[4]->eval();
        if(any_of({gamma, bias, mean, variance}, [](auto arg) { return arg.empty(); }))
            continue;

        std::vector<std::size_t> lens = ins->inputs()[1]->get_shape().lens();
        shape s{ins->get_shape().type(), lens};
        // Get epsilon
        auto bn_op   = any_cast<op::batch_norm_inference>(ins->get_operator());
        auto epsilon = bn_op.epsilon;

        argument a{s};
        argument b{s};
        visit_all(gamma, bias, mean, variance, a, b)(
            [&](auto gamma2, auto bias2, auto mean2, auto variance2, auto a2, auto b2) {
                dfor(a.get_shape().elements())(
                    [&](std::size_t c) { a2[c] = gamma2[c] / std::sqrt(variance2[c] + epsilon); });
                dfor(b.get_shape().elements())([&](std::size_t c) {
                    b2[c] = bias2[c] - (gamma2[c] * mean2[c] / std::sqrt(variance2[c] + epsilon));
                });
            });

        auto broadcast   = op::broadcast{1, ins->get_shape().lens()};
        auto a_ins       = m.add_literal({a.get_shape(), a.data()});
        auto a_broadcast = m.insert_instruction(ins, broadcast, a_ins);
        auto mul   = m.insert_instruction(ins, make_op("mul"), ins->inputs().front(), a_broadcast);
        auto b_ins = m.add_literal({b.get_shape(), b.data()});
        auto b_broadcast = m.insert_instruction(ins, broadcast, b_ins);
        auto add         = m.insert_instruction(ins, make_op("add"), mul, b_broadcast);
        m.replace_instruction(ins, add);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
