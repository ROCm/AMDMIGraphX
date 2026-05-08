/*
 * The MIT License (MIT)
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
#include <migraphx/fast_mm.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void fast_mm::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(not contains({"convolution", "quant_convolution"}, ins->name()))
            continue;

        const auto out_type = ins->get_shape().type();
        if(out_type != shape::float_type)
            continue;

        auto inputs = ins->inputs();
        if(std::any_of(inputs.begin(), inputs.end(), [](auto input) {
               return input->get_shape().type() != shape::float_type;
           }))
            continue;

        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto input) {
            return m.insert_instruction(
                ins, make_op("convert", {{"target_type", shape::half_type}}), input);
        });

        auto half_conv = m.insert_instruction(ins, ins->get_operator(), inputs);
        auto converted =
            m.insert_instruction(ins, make_op("convert", {{"target_type", out_type}}), half_conv);
        m.replace_instruction(ins, converted);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
