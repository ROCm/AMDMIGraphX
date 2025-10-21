/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/eliminate_zero_dim.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/make_op.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void eliminate_zero_dim::apply(module& m) const
{
    auto last = std::prev(m.end());
    for(auto ins : iterator_for(m))
    {
        if(ins == m.begin())
            continue;
        const auto i = std::prev(ins);
        if(i->name() != "concat" or i->inputs().size() !=  2)
            continue;

        auto inputs = i->inputs();
        auto cntr = 0;
        auto idx0 = cntr;
        auto idx = cntr;
        if(std::none_of(inputs.begin(), inputs.end(), [&](auto inp){
            auto lens = inp->get_shape().lens();
            bool has_zero_dim = std::any_of(lens.begin(), lens.end(), [](auto l){ return l == 0; });
            if(has_zero_dim)
            {
                idx0 = cntr;
            }
            else    
            {
                idx = cntr;
            }
            cntr++;
            auto is_convert = inp->name() == "convert";
            return has_zero_dim and is_convert;
        }))
            continue;

        idx = idx0 + 1 % 2;
        auto new_convert = m.insert_instruction(i, make_op("convert", {{"target_type", inputs[idx0]->inputs()[0]->get_shape().type()}}), inputs[idx]);
        auto new_inputs = inputs;
        new_inputs[idx0] = inputs[idx0]->inputs()[0];
        new_inputs[idx]  = new_convert;
        auto new_concat = m.insert_instruction(i, make_op("concat", {{"axis", i->get_operator().attributes().get("axis", 2)}}), new_inputs);
        m.replace_instruction(i, make_op("convert", {{"target_type", inputs[idx]->get_shape().type()}}), new_concat);
        m.remove_instruction(inputs[idx0]);


    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
