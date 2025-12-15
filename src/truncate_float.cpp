/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/float_equal.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/truncate_float.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/target.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static void
quantize_module(module& m, const std::vector<std::string>& ins_names, shape::type_t float_type)
{
    for(auto ins : iterator_for(m))
    {
        // instructions are not in the set to be quantized
        if(not(contains(ins_names, ins->name()) or contains(ins_names, "all")))
            continue;

        // skip return and convert instructions
        if(contains({"@return", "convert"}, ins->name()))
            continue;

        if(ins->inputs().empty())
            continue;

        auto mod_inputs = ins->module_inputs();
        auto s          = ins->get_shape();

        // Skip instructions with 0 elements or any 0-element inputs - they can cause
        // issues in JIT pointwise kernel compilation due to size_t underflow in
        // compute_global_for. We must skip the entire instruction to avoid type
        // mismatches (e.g., in concat where some inputs would be converted and others not).
        // Note: tuple shapes return 0 for elements() but should not be skipped.
        if(not s.dynamic() and s.type() != shape::tuple_type and s.elements() == 0)
            continue;
        auto inputs = ins->inputs();
        bool has_zero_element_input = std::any_of(inputs.begin(), inputs.end(), [](auto input) {
            const auto& input_shape = input->get_shape();
            return not input_shape.dynamic() and input_shape.type() != shape::tuple_type and
                   input_shape.elements() == 0;
        });
        if(has_zero_element_input)
            continue;

        // Convert each of the inputs that are floating point to float type
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto input) {
            auto input_type = input->get_shape().type();
            if(input_type != shape::float_type and input_type != shape::double_type)
                return input;
            return m.insert_instruction(
                ins, make_op("convert", {{"target_type", float_type}}), input);
        });

        // Insert quantized ins
        auto converted_ins = m.insert_instruction(ins, ins->get_operator(), inputs, mod_inputs);

        // tuple can't be directly converted, get_tuple_elem needs conversion
        if(ins->get_shape().type() == shape::tuple_type)
        {
            auto outputs = ins->outputs();
            std::transform(
                outputs.begin(), outputs.end(), outputs.begin(), [&](const auto gte_ins) {
                    auto gte_ins_float_type =
                        m.insert_instruction(ins, gte_ins->get_operator(), converted_ins);
                    // Convert back to output type after quantizing
                    auto gte_converted = m.insert_instruction(
                        ins,
                        make_op("convert", {{"target_type", gte_ins->get_shape().type()}}),
                        gte_ins_float_type);
                    // Replace output instruction
                    return m.replace_instruction(gte_ins, gte_converted);
                });
        }
        else
        {
            // Convert back to original type after quantizing
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

void truncate_float_pass::apply(module& m) const { quantize_module(m, ins_names, float_type); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
