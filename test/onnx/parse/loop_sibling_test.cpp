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

TEST_CASE(loop_sibling_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape si{migraphx::shape::int64_type, {1}};
    auto max_iter = mm->add_parameter("max_trip_count", si);
    migraphx::shape sc{migraphx::shape::bool_type, {1}};
    auto icond = mm->add_parameter("keep_going_cond", sc);
    migraphx::shape su{migraphx::shape::float_type, {1}};
    auto a = mm->add_parameter("a", su);
    auto b = mm->add_parameter("b", su);

    auto* body1 = p.create_module("main_Loop_4_loop");
    body1->add_parameter("iteration_num", si);
    auto cond1 = body1->add_parameter("keep_going_inp", sc);
    auto var1  = body1->add_parameter("b_in", su);
    auto id1   = body1->add_instruction(migraphx::make_op("identity"), cond1);
    auto ad1   = body1->add_instruction(migraphx::make_op("add"), a, var1);
    body1->add_return({id1, ad1});
    auto lp1 = mm->add_instruction(
        migraphx::make_op("loop", {{"max_iterations", 10}}), {max_iter, icond, b}, {body1});
    auto r1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), lp1);

    // The second body uses the same parameter names as the first one
    auto* body2 = p.create_module("main_Loop_6_loop");
    body2->add_parameter("iteration_num", si);
    auto cond2 = body2->add_parameter("keep_going_inp", sc);
    auto var2  = body2->add_parameter("b_in", su);
    auto id2   = body2->add_instruction(migraphx::make_op("identity"), cond2);
    auto ad2   = body2->add_instruction(migraphx::make_op("add"), a, var2);
    body2->add_return({id2, ad2});
    auto lp2 = mm->add_instruction(
        migraphx::make_op("loop", {{"max_iterations", 10}}), {max_iter, icond, r1}, {body2});
    auto r2 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), lp2);
    mm->add_return({r2});

    auto prog = read_onnx("loop_sibling_test.onnx");

    EXPECT(p == prog);
}
