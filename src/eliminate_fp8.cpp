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
#include "migraphx/serialize.hpp"
#include <iterator>
#include <utility>
#include <migraphx/eliminate_fp8.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void eliminate_fp8::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(not contains(op_names, ins->name()) or
           ins->get_shape().type() != migraphx::shape::fp8e4m3fnuz_type)
            continue;
        migraphx::shape::type_t orig_type        = ins->get_shape().type();
        std::vector<instruction_ref> orig_inputs = ins->inputs();
        std::vector<instruction_ref> new_inputs;
        std::transform(orig_inputs.begin(),
                       orig_inputs.end(),
                       std::back_inserter(new_inputs),
                       [&](const auto& i) {
                           return m.insert_instruction(
                               ins,
                               migraphx::make_op(
                                   "convert", {{"target_type", migraphx::to_value(target_type)}}),
                               i);
                       });

        auto new_ins          = m.insert_instruction(ins, ins->get_operator(), {new_inputs});
        auto convert_back_ins = m.insert_instruction(
            ins,
            migraphx::make_op("convert", {{"target_type", migraphx::to_value(orig_type)}}),
            new_ins);
        m.replace_instruction(ins, convert_back_ins);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
