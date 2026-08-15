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
#include <onnx_test.hpp>

// Regression test: parsing Softsign with a non-fixed dynamic dimension used to abort.
// Same root cause as softplus_dyn_test - parse_softsign built its broadcast literal with
// multibroadcast{"out_lens", args[0]->get_shape().lens()}, and shape::lens() throws on a
// dynamic shape. See softplus_dyn_test for the full explanation.
TEST_CASE(softsign_dyn_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto input_type = migraphx::shape::float_type;
    // Non-fixed first dimension: this is what used to throw.
    migraphx::shape s{input_type, {{1, 4}, {5, 5}}};

    auto x    = mm->add_parameter("x", s);
    auto ones = mm->add_literal(migraphx::literal{migraphx::shape{input_type}, {1}});
    auto abs  = mm->add_instruction(migraphx::make_op("abs"), x);
    // See softplus_dyn_test: add_common_op broadcasts both operands, so two multibroadcasts,
    // each in the two-input dynamic-safe form with the resolved dims in out_dyn_dims.
    auto mb_abs = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_dyn_dims", to_value(s.dyn_dims())}}),
        abs,
        ones);
    auto mb_ones = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_dyn_dims", to_value(s.dyn_dims())}}),
        ones,
        mb_abs);
    auto add = mm->add_instruction(migraphx::make_op("add"), mb_abs, mb_ones);
    auto r       = mm->add_instruction(migraphx::make_op("div"), x, add);
    mm->add_return({r});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["x"] = {{1, 4}, {5, 5}};
    auto prog                       = read_onnx("softsign_test.onnx", options);
    EXPECT(p == prog);
}
