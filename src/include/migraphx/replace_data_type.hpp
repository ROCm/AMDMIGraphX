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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_REPLACE_DATA_TYPE_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_REPLACE_DATA_TYPE_HPP

#include <migraphx/config.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape.hpp>
#include <algorithm>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/// Compute each instruction matching the predicate in target_type instead:
/// inputs of a src type are converted to target_type and the result is
/// converted back to the original type.
template <class Predicate>
void replace_data_type(module& m,
                       const std::vector<shape::type_t>& src,
                       shape::type_t target_type,
                       Predicate predicate)
{
    for(auto ins : iterator_for(m))
    {
        if(not predicate(ins))
            continue;

        // skip return and convert instructions
        if(contains({"@return", "convert"}, ins->name()))
            continue;

        if(ins->inputs().empty())
            continue;

        auto mod_inputs = ins->module_inputs();
        auto s          = ins->get_shape();
        // Convert each of the src type inputs to the target type
        auto inputs = ins->inputs();
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto input) {
            auto input_type = input->get_shape().type();
            if(not contains(src, input_type))
                return input;
            return m.insert_instruction(
                ins, make_op("convert", {{"target_type", target_type}}), input);
        });

        // Insert replaced ins
        auto converted_ins = m.insert_instruction(ins, ins->get_operator(), inputs, mod_inputs);

        // tuple can't be directly converted, get_tuple_elem needs conversion
        if(ins->get_shape().type() == shape::tuple_type)
        {
            auto outputs = ins->outputs();
            std::transform(
                outputs.begin(), outputs.end(), outputs.begin(), [&](const auto gte_ins) {
                    auto gte_ins_target_type =
                        m.insert_instruction(ins, gte_ins->get_operator(), converted_ins);
                    // Convert back to the original type
                    auto gte_converted = m.insert_instruction(
                        ins,
                        make_op("convert", {{"target_type", gte_ins->get_shape().type()}}),
                        gte_ins_target_type);
                    // Replace output instruction
                    return m.replace_instruction(gte_ins, gte_converted);
                });
        }
        else
        {
            // Convert back to the original type
            if(mod_inputs.empty())
            {
                converted_ins = m.insert_instruction(
                    ins, make_op("convert", {{"target_type", s.type()}}), converted_ins);
            }
            // Replace original instruction
            m.replace_instruction(ins, converted_ins);
        }
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_REPLACE_DATA_TYPE_HPP
