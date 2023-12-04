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
#include <iterator>
#include <utility>
#include <migraphx/eliminate_fp8.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void eliminate_fp8::apply(module& m) const
{
    for(auto ins : iterator_for(m))
    {
        if(not contains(op_names, ins->name()))
            continue;
        migraphx::shape::type_t orig_type   = ins->get_shape().type();
        std::vector<instruction_ref> inputs = ins->inputs();
        migraphx::transform_if(
            inputs.begin(),
            inputs.end(),
            inputs.begin(),
            [&](const auto& i) { return i->get_shape().type() == shape::fp8e4m3fnuz_type; },
            [&](const auto& i) {
                return m.insert_instruction(
                    ins,
                    migraphx::make_op("convert",
                                      {{"target_type", migraphx::to_value(target_type)}}),
                    i);
            });
        if(inputs == ins->inputs())
        {
            return;
        }
        auto op         = ins->get_operator();
        auto attributes = op.attributes();
        if(attributes.contains("general_data_type"))
        {
            op = make_op(attributes["general_data_type"].to<std::string>(), op.to_value());
        }
        auto new_ins = m.insert_instruction(ins, op, inputs);
        if(orig_type == shape::tuple_type)
        {
            auto orig_outs = ins->outputs();
            if(not std::all_of(orig_outs.begin(), orig_outs.end(), [&](const auto out_ins) {
                   return out_ins->name() == "get_tuple_elem";
               }))
                MIGRAPHX_THROW("EliminateFP8: Instruction with tuple output doesn't have all its "
                               "usages as get_tuple_elem instruction");

            std::transform(
                orig_outs.begin(), orig_outs.end(), orig_outs.begin(), [&](const auto out_ins) {
                    auto gte_ins = m.insert_instruction(ins, out_ins->get_operator(), new_ins);
                    if(out_ins->get_shape().type() == shape::type_t::fp8e4m3fnuz_type)
                    {
                        auto gte_convert = m.insert_instruction(
                            ins,
                            make_op("convert", {{"target_type", shape::type_t::fp8e4m3fnuz_type}}),
                            gte_ins);
                        return m.replace_instruction(out_ins, gte_convert);
                    }
                    else
                    {
                        return m.replace_instruction(out_ins, gte_ins);
                    }
                });
        }
        else
        {
            auto convert_back_ins = m.insert_instruction(
                ins,
                migraphx::make_op("convert", {{"target_type", migraphx::to_value(orig_type)}}),
                new_ins);
            m.replace_instruction(ins, convert_back_ins);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
