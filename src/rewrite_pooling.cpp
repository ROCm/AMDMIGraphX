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
#include <migraphx/rewrite_pooling.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/op/pooling.hpp>
#include <migraphx/op/reshape.hpp>
#include <migraphx/op/reduce_mean.hpp>
#include <migraphx/op/reduce_max.hpp>
#include <migraphx/make_op.hpp>

#include <migraphx/program.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void rewrite_pooling::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "pooling")
            continue;
        if(ins->inputs().empty())
            continue;
        auto&& s = ins->inputs().front()->get_shape();
        if(not s.standard())
            continue;
        auto&& op = any_cast<op::pooling>(ins->get_operator());
        if(not std::all_of(op.padding.begin(), op.padding.end(), [](auto i) { return i == 0; }))
            continue;
        if(not std::all_of(op.stride.begin(), op.stride.end(), [](auto i) { return i == 1; }))
            continue;
        auto lens = s.lens();
        if(not std::equal(lens.begin() + 2, lens.end(), op.lengths.begin(), op.lengths.end()))
            continue;
        std::int64_t n = s.lens()[0];
        std::int64_t c = s.lens()[1];
        auto reshape   = m.insert_instruction(
            ins, make_op("reshape", {{"dims", {n * c, -1}}}), ins->inputs().front());
        instruction_ref pooling{};

        // average pooling
        if(op.mode == op::pooling_mode::average)
        {
            pooling = m.insert_instruction(ins, make_op("reduce_mean", {{"axes", {1}}}), reshape);
        }
        // max pooling
        else
        {
            pooling = m.insert_instruction(ins, make_op("reduce_max", {{"axes", {1}}}), reshape);
        }

        std::vector<int64_t> rsp_lens(lens.size(), 1);
        rsp_lens[0] = n;
        rsp_lens[1] = c;
        m.replace_instruction(ins, make_op("reshape", {{"dims", rsp_lens}}), pooling);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
