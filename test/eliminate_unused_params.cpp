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
#include <migraphx/eliminate_unused_params.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <test.hpp>
#include <pointwise.hpp>

static void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p,
                         {migraphx::eliminate_unused_params{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(removes_unused_pointwise_param)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        // y is unused in the body, the general shape left by a fold (e.g. alpha*A*B
        // when alpha == 0) that drops a submodule parameter entirely.
        add_pointwise(p1, mm, "main:pointwise0", {x, y}, [=](auto* pm, const auto& params) {
            return pm->add_instruction(migraphx::make_op("identity"), params[0]);
        });
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        mm->add_parameter("y", s);
        add_pointwise(p2, mm, "main:pointwise0:used", {x}, [=](auto* pm, const auto& params) {
            return pm->add_instruction(migraphx::make_op("identity"), params[0]);
        });
    }
    EXPECT(p1 == p2);
}

TEST_CASE(keeps_pointwise_with_all_params_used)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        add_pointwise(p1, mm, "main:pointwise0", {x, y}, [=](auto* pm, const auto& params) {
            return pm->add_instruction(migraphx::make_op("add"), params[0], params[1]);
        });
    }

    migraphx::program p2 = p1;
    run_pass(p2);

    EXPECT(p1 == p2);
}

// This body has an unused parameter and the same parameter/operand count as the loop
// instruction, so a count-based check alone would not rule it out.
TEST_CASE(keeps_unused_loop_body_param)
{
    migraphx::shape si{migraphx::shape::int64_type};
    migraphx::shape s{migraphx::shape::int64_type, {1}};
    migraphx::shape sc{migraphx::shape::bool_type};

    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto in_cond = mm->add_parameter("ccond", sc);
        auto in_val  = mm->add_parameter("val", s);

        auto* body = p1.create_module("loop_module");
        body->add_parameter("#loop_module_in_1", sc);
        auto in_v = body->add_parameter("#loop_module_in_2", s);
        auto lit  = body->add_literal(migraphx::literal(si, {3}));
        auto val  = body->add_instruction(migraphx::make_op("add"), in_v, lit);
        auto cond = body->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), in_cond);
        body->add_return({cond, val, val});

        auto loop = mm->add_instruction(
            migraphx::make_op("loop", {{"max_iterations", 1}}), {in_cond, in_val}, {body});
        mm->add_return({loop});
    }

    migraphx::program p2 = p1;
    run_pass(p2);

    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
