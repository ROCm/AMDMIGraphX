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

TEST_CASE(loop_outer_scope_nested_unordered_test)
{
    // The Loop node is listed before the Add producing a2, which an If inside its body reads
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape si{migraphx::shape::int64_type, {1}};
    auto max_iter = mm->add_parameter("max_trip_count", si);
    migraphx::shape sc{migraphx::shape::bool_type, {1}};
    auto icond = mm->add_parameter("keep_going_cond", sc);
    migraphx::shape su{migraphx::shape::float_type, {1}};
    auto a  = mm->add_parameter("a", su);
    auto b  = mm->add_parameter("b", su);
    auto a2 = mm->add_instruction(migraphx::make_op("add"), a, a);

    auto* body = p.create_module("main_Loop_5_loop");
    body->add_parameter("iteration_num", si);
    auto cond = body->add_parameter("keep_going_inp", sc);
    auto var  = body->add_parameter("b_in", su);
    auto id   = body->add_instruction(migraphx::make_op("identity"), cond);

    auto* then_mod = p.create_module("main_Loop_5_loop_If_4_if");
    auto then_add  = then_mod->add_instruction(migraphx::make_op("add"), a2, var);
    then_mod->add_return({then_add});
    auto* else_mod = p.create_module("main_Loop_5_loop_If_4_else");
    auto else_sub  = else_mod->add_instruction(migraphx::make_op("sub"), a2, var);
    else_mod->add_return({else_sub});

    auto ret = body->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
    auto r   = body->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
    body->add_return({id, r});

    auto lp = mm->add_instruction(
        migraphx::make_op("loop", {{"max_iterations", 10}}), {max_iter, icond, b}, {body});
    auto r0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), lp);
    mm->add_return({r0});

    auto prog = read_onnx("loop_outer_scope_nested_unordered_test.onnx");

    EXPECT(p == prog);
}
