/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/eliminate_data_type.hpp>
#include <migraphx/module.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void eliminate_data_type::apply(module& m) const
{
    static const std::vector<std::string> skip_op_names = {"convert",
                                                           "get_tuple_elem",
                                                           "if",
                                                           "loop",
                                                           "roialign",
                                                           "nonmaxsuppression",
                                                           "scatternd_add",
                                                           "scatternd_mul",
                                                           "scatternd_none"};
    for(auto ins : iterator_for(m))
    {
        if(ins->name()[0] == '@')
            continue;
        if(contains(skip_op_names, ins->name()))
            continue;
        auto inputs = ins->inputs();
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto i) {
            if(types.count(i->get_shape().type()) == 0)
                return i;
            return m.insert_instruction(ins, make_op("convert", {{"target_type", target_type}}), i);
        });
        if(inputs == ins->inputs())
            continue;
        auto op         = ins->get_operator();
        auto attributes = op.attributes();
        if(attributes.contains("general_data_type"))
        {
            op = make_op(attributes["general_data_type"].to<std::string>(), op.to_value());
        }
        auto old_type = ins->get_shape().type();
        auto out      = m.insert_instruction(ins, op, inputs);
        auto convert =
            m.insert_instruction(ins, make_op("convert", {{"target_type", old_type}}), out);
        m.replace_instruction(ins, convert);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
