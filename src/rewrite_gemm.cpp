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

#include <migraphx/rewrite_gemm.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/serialize.hpp>

#include <migraphx/make_op.hpp>

#include <migraphx/pass_config.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void rewrite_gemm::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "dot")
            continue;
        auto inputs = ins->inputs();
        auto in0    = inputs.at(0);
        if(in0->get_shape().lens().at(0) != 1) // only batch size = 1
            continue;
        auto in1 = inputs.at(1);

        auto in0_transposed =
            m.insert_instruction(ins, make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), in0);
        auto in1_transposed =
            m.insert_instruction(ins, make_op("transpose", {{"permutation", {3, 2, 1, 0}}}), in1);

        auto conv     = make_op("convolution");
        auto conv_out = m.replace_instruction(ins, conv, {in0_transposed, in1_transposed});
        auto conv_transpose =
            m.add_instruction(make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), conv_out);
        // m.insert_instruction(std::next(conv_transpose), make_op("unsqueeze"));
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
