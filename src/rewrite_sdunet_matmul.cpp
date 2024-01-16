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

#include <migraphx/rewrite_sdunet_matmul.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/common.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct find_sdunet_matmul
{
    auto dot() const { return match::name("dot").bind("dot"); }

    auto mul() const
    {
        return match::name("mul")(match::arg(0)(dot()),
                                  match::arg(1)(match::any().bind("mul_right")))
            .bind("mul");
    }

    auto softmax() const { return match::name("softmax")(match::arg(0)(mul())).bind("softmax"); }

    auto dot2() const
    {
        return match::name("dot")(match::arg(0)(softmax()),
                                  match::arg(1)(match::any().bind("dot_right")))
            .bind("dot2");
    }

    auto matcher() const { return dot2(); }

    void apply(module& m, const match::matcher_result& r) const
    {
        auto dot = r.instructions["dot"];
        // We only need to run when fp16 is used
        if(dot->get_shape().type() != migraphx::shape::half_type)
            return;
        std::vector<instruction_ref> dot_inputs;
        dot_inputs.emplace_back(m.insert_instruction(
            dot, make_op("convert", {{"target_type", shape::float_type}}), dot->inputs().at(0)));
        dot_inputs.emplace_back(m.insert_instruction(
            dot, make_op("convert", {{"target_type", shape::float_type}}), dot->inputs().at(1)));
        auto converted_dot = m.insert_instruction(dot, dot->get_operator(), dot_inputs);

        auto mul = r.instructions["mul"];
        std::vector<instruction_ref> mul_inputs;
        mul_inputs.emplace_back(m.insert_instruction(
            mul, make_op("convert", {{"target_type", shape::float_type}}), converted_dot));
        mul_inputs.emplace_back(
            m.insert_instruction(mul,
                                 make_op("convert", {{"target_type", shape::float_type}}),
                                 r.instructions["mul_right"]));
        auto converted_mul = m.insert_instruction(mul, mul->get_operator(), mul_inputs);

        auto softmax = r.instructions["softmax"];
        auto converted_softmax =
            m.insert_instruction(softmax, softmax->get_operator(), converted_mul);

        auto dot2 = r.instructions["dot2"];
        std::vector<instruction_ref> dot2_inputs;
        dot2_inputs.emplace_back(m.insert_instruction(
            dot2, make_op("convert", {{"target_type", shape::half_type}}), converted_softmax));
        dot2_inputs.emplace_back(r.instructions["dot_right"]);
        auto converted_dot2 = m.insert_instruction(dot2, dot2->get_operator(), dot2_inputs);

        m.replace_instruction(dot2, converted_dot2);
        // cleanup
        m.remove_instruction(dot);
        m.remove_instruction(mul);
        m.remove_instruction(softmax);
        m.remove_instruction(dot2);
    }
};

void rewrite_sdunet_matmul::apply(module& m) const { match::find_matches(m, find_sdunet_matmul{}); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
