/*
 * The MIT License (MIT)
 ,*
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
#include <migraphx/fp_to_double.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/matcher.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

void module_fp_to_double(module& m)
{
    for(auto ins : iterator_for(m))
    {
        // skip return and convert instructions
        if(contains({"@return", "convert"}, ins->name()))
            continue;

        if(ins->inputs().empty())
            continue;

        auto mod_inputs = ins->module_inputs();
        auto s          = ins->get_shape();
        // Convert each of the inputs that are floating point to double
        auto inputs = ins->inputs();
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto input) {
            auto input_type = input->get_shape().type();
            if(input_type != shape::half_type and input_type != shape::float_type)
                return input;
            return m.insert_instruction(
                ins, make_op("convert", {{"target_type", shape::double_type}}), input);
        });
        // Insert new ins
        auto converted_ins = m.insert_instruction(ins, ins->get_operator(), inputs, mod_inputs);

        // Convert back to original type afterward
        if(mod_inputs.empty())
        {
            converted_ins = m.insert_instruction(
                ins, make_op("convert", {{"target_type", s.type()}}), converted_ins);
        }
        // Replace original instruction
        m.replace_instruction(ins, converted_ins);
    }
}

struct find_nested_convert
{
    auto matcher() const { return match::name("convert")(match::arg(0)(match::name("convert"))); }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins   = mr.result;
        auto x     = ins->inputs().front();
        auto input = x->inputs().front();

        if(ins->get_shape() != input->get_shape())
            return;

        m.replace_instruction(ins, input);
    }
};

struct find_nop_converts
{
    auto matcher() const { return match::name("convert")(match::same_shape(match::arg(0))); }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins = mr.result;
        m.replace_instruction(ins, ins->inputs().front());
    }
};

void fp_to_double::apply(module& m) const
{
    module_fp_to_double(m);
    match::find_matches(m, find_nested_convert{}, find_nop_converts{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
