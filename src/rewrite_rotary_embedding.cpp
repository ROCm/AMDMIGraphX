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
#include <migraphx/rewrite_rotary_embedding.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/module.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/value.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/literal.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

struct find_rotary_embedding
{
    auto matcher() const
    {
        auto input = match::any().bind("input");
        auto neg = match::name("neg")(match::arg(0)(match::name("slice")(match::arg(0)(input)).bind("slice1"))).bind("neg");
        auto slice = match::name("slice")(match::arg(0)(input)).bind("slice0");
        auto concat = match::name("concat")(match::arg(0)(neg), match::arg(1)(slice));
        auto mul_sin = match::name("mul")(match::arg(0)(concat), match::arg(1)(match::is_constant().bind("sin")));
        auto mul_cos = match::name("mul")(match::arg(0)(input), match::arg(1)(match::is_constant())).bind("mul_cos");
        return match::name("add")(match::either_arg(0, 1)(mul_sin, mul_cos));
    }

    void apply(module_pass_manager& mpm, const match::matcher_result& r) const
    {
        auto ins = r.result;
        auto input = r.instructions["input"];
        auto slice1 = r.instructions["slice1"];
        auto slice0 = r.instructions["slice0"];
        auto mul_cos = r.instructions["mul_cos"];
        auto sin = r.instructions["sin"];

        auto axes = slice0->get_operator().to_value()["axes"].template to_vector<int64_t>();
        if (axes.size() != 1)
            return;
        auto axis = axes[0];

        auto slice0_lens = slice0->get_shape().lens();
        auto slice1_lens = slice1->get_shape().lens();
        auto input_lens = input->get_shape().lens();
        auto d = input_lens[axis];
        if (d != slice0_lens[axis] + slice1_lens[axis])
            return;
        auto half_d = slice0_lens[axis];
        auto signs = mpm.get_module().add_literal(literal{shape{input->get_shape().type(), {2}}, {-1.0f, 1.0f}});
        signs = mpm.get_module().insert_instruction(ins, make_op("reshape", {{"dims", {2, 1}}}), signs);
        signs = mpm.get_module().insert_instruction(ins, make_op("multibroadcast", {{"out_lens", {2, half_d}}}), signs);
        signs = mpm.get_module().insert_instruction(ins, make_op("reshape", {{"dims", {d}}}), signs);
        signs = mpm.get_module().insert_instruction(ins, make_op("multibroadcast", {{"out_lens", input_lens}}), signs);

        auto concat = mpm.get_module().insert_instruction(ins, make_op("concat", {{"axis", axis}}), slice1, slice0);
        auto mul_sin = mpm.get_module().insert_instruction(ins, make_op("mul"), signs, sin);
        mul_sin = mpm.get_module().insert_instruction(ins, make_op("mul"), mul_sin, concat);
        auto add = mpm.get_module().insert_instruction(ins, make_op("add"), mul_cos, mul_sin);
        mpm.get_module().replace_instruction(ins, add);
    }
};

} // namespace

void rewrite_rotary_embedding::apply(module_pass_manager& mpm) const
{
    match::find_matches(mpm, find_rotary_embedding{});
    mpm.run_pass(dead_code_elimination{});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
