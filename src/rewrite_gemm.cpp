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

        if(in0->get_shape().lens().size() > 2)
            if(in0->get_shape().lens().at(0) != 1) // only batch size = 1
                continue;
        auto in_size = in0->get_shape().lens().size();
        if(in_size == 4 and in0->get_shape().lens().at(1) != 1)
        {
            continue;
        }
        auto in1 = inputs.at(1);

        if(in_size < 4)
        {
            std::vector<size_t> new_lens0(in0->get_shape().lens().begin(),
                                          in0->get_shape().lens().end());
            std::vector<size_t> new_lens1(in1->get_shape().lens().begin(),
                                          in1->get_shape().lens().end());
            std::vector<size_t> ones(4 - in_size, 1);
            new_lens0.insert(new_lens0.begin(), ones.begin(), ones.end());
            new_lens1.insert(new_lens1.begin(), ones.begin(), ones.end());
            if(not in0->get_shape().standard())
                in0 = m.insert_instruction(ins, make_op("contiguous"), in0);
            in0 = m.insert_instruction(ins, make_op("reshape", {{"dims", new_lens0}}), in0);
            if(not in1->get_shape().standard())
                in1 = m.insert_instruction(ins, make_op("contiguous"), in1);
            in1 = m.insert_instruction(ins, make_op("reshape", {{"dims", new_lens1}}), in1);
        }

        auto in0_transposed =
            m.insert_instruction(ins, make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), in0);
        auto in1_transposed =
            m.insert_instruction(ins, make_op("transpose", {{"permutation", {3, 2, 1, 0}}}), in1);

        auto conv      = m.insert_instruction(ins, make_op("convolution"), {in0_transposed, in1_transposed});
        auto conv_transpose = m.insert_instruction(
            ins, make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), conv);

        auto out_lens = conv_transpose->get_shape().lens();
        auto conv_transpose_out = conv_transpose;
        if(out_lens.size() != in_size)
        {
            out_lens.erase(out_lens.begin(), out_lens.begin() + (out_lens.size() - in_size));
            conv_transpose_out = m.insert_instruction(ins,
                                 make_op("reshape", {{"dims", out_lens}}),
                                 conv_transpose);
        }
        m.replace_instruction(ins, conv_transpose_out);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
