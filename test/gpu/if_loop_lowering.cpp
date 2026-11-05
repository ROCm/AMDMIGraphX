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
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/lowering.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>

static void run_lowering(migraphx::module& m)
{
    auto ctx = migraphx::gpu::context{};
    migraphx::run_passes(m,
                         {migraphx::gpu::lowering{&ctx, false}, migraphx::dead_code_elimination{}});
}

TEST_CASE(if_lowering_load_scalar)
{
    migraphx::shape cond_s{migraphx::shape::bool_type, {1}};
    migraphx::shape s{migraphx::shape::float_type, {2}};

    migraphx::program p1;
    {
        auto* mm       = p1.get_main_module();
        auto cond      = mm->add_parameter("cond", cond_s);
        auto* then_mod = p1.create_module("If_0_if");
        then_mod->add_return({then_mod->add_literal(migraphx::literal{s, {1.0f, 2.0f}})});
        auto* else_mod = p1.create_module("If_0_else");
        else_mod->add_return({else_mod->add_literal(migraphx::literal{s, {3.0f, 4.0f}})});
        auto r = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return({r});
    }
    run_lowering(*p1.get_main_module());

    migraphx::program p2;
    {
        auto* mm       = p2.get_main_module();
        auto cond      = mm->add_parameter("cond", cond_s);
        auto* then_mod = p2.create_module("If_0_if");
        then_mod->add_return({then_mod->add_literal(migraphx::literal{s, {1.0f, 2.0f}})});
        auto* else_mod = p2.create_module("If_0_else");
        else_mod->add_return({else_mod->add_literal(migraphx::literal{s, {3.0f, 4.0f}})});
        auto loaded = mm->add_instruction(migraphx::make_op("hip::load_scalar"), cond);
        auto r      = mm->add_instruction(migraphx::make_op("if"), {loaded}, {then_mod, else_mod});
        mm->add_return({r});
    }
    EXPECT(p1 == p2);
}

// hip::load_scalar keeps the stride-0 scalar shape of the condition.
TEST_CASE(if_lowering_scalar_cond)
{
    migraphx::shape cond_s{migraphx::shape::bool_type};
    migraphx::shape s{migraphx::shape::float_type, {2}};

    migraphx::program p1;
    {
        auto* mm       = p1.get_main_module();
        auto cond      = mm->add_parameter("cond", cond_s);
        auto* then_mod = p1.create_module("If_0_if");
        then_mod->add_return({then_mod->add_literal(migraphx::literal{s, {1.0f, 2.0f}})});
        auto* else_mod = p1.create_module("If_0_else");
        else_mod->add_return({else_mod->add_literal(migraphx::literal{s, {3.0f, 4.0f}})});
        auto r = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return({r});
    }
    run_lowering(*p1.get_main_module());

    migraphx::program p2;
    {
        auto* mm       = p2.get_main_module();
        auto cond      = mm->add_parameter("cond", cond_s);
        auto* then_mod = p2.create_module("If_0_if");
        then_mod->add_return({then_mod->add_literal(migraphx::literal{s, {1.0f, 2.0f}})});
        auto* else_mod = p2.create_module("If_0_else");
        else_mod->add_return({else_mod->add_literal(migraphx::literal{s, {3.0f, 4.0f}})});
        auto loaded = mm->add_instruction(migraphx::make_op("hip::load_scalar"), cond);
        EXPECT(loaded->get_shape() == cond_s);
        auto r = mm->add_instruction(migraphx::make_op("if"), {loaded}, {then_mod, else_mod});
        mm->add_return({r});
    }
    EXPECT(p1 == p2);
}

// hip::load_scalar only takes one element, so a multi-element condition keeps copy_from_gpu.
TEST_CASE(if_lowering_multi_element_cond)
{
    migraphx::shape cond_s{migraphx::shape::bool_type, {2}};
    migraphx::shape s{migraphx::shape::float_type, {2}};

    migraphx::program p1;
    {
        auto* mm       = p1.get_main_module();
        auto cond      = mm->add_parameter("cond", cond_s);
        auto* then_mod = p1.create_module("If_0_if");
        then_mod->add_return({then_mod->add_literal(migraphx::literal{s, {1.0f, 2.0f}})});
        auto* else_mod = p1.create_module("If_0_else");
        else_mod->add_return({else_mod->add_literal(migraphx::literal{s, {3.0f, 4.0f}})});
        auto r = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        mm->add_return({r});
    }
    run_lowering(*p1.get_main_module());

    migraphx::program p2;
    {
        auto* mm       = p2.get_main_module();
        auto cond      = mm->add_parameter("cond", cond_s);
        auto* then_mod = p2.create_module("If_0_if");
        then_mod->add_return({then_mod->add_literal(migraphx::literal{s, {1.0f, 2.0f}})});
        auto* else_mod = p2.create_module("If_0_else");
        else_mod->add_return({else_mod->add_literal(migraphx::literal{s, {3.0f, 4.0f}})});
        auto cpu_cond = mm->add_instruction(migraphx::make_op("hip::copy_from_gpu"), cond);
        auto synced   = mm->add_instruction(migraphx::make_op("hip::sync_stream"), cpu_cond);
        auto r = mm->add_instruction(migraphx::make_op("if"), {synced}, {then_mod, else_mod});
        mm->add_return({r});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(loop_lowering_load_scalar)
{
    migraphx::shape iter_s{migraphx::shape::int64_type};
    migraphx::shape cond_s{migraphx::shape::bool_type};
    migraphx::shape x_s{migraphx::shape::float_type, {2}};
    migraphx::shape out_s{std::vector<migraphx::shape>{x_s}};
    auto loop = migraphx::make_op("loop", {{"max_iterations", 3}});

    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto iter  = mm->add_parameter("iter", iter_s);
        auto cond  = mm->add_parameter("cond", cond_s);
        auto x     = mm->add_parameter("x", x_s);
        auto* body = p1.create_module("Loop_0_body");
        body->add_parameter("iter_num", iter_s);
        auto body_cond = body->add_parameter("cond", cond_s);
        auto body_x    = body->add_parameter("x", x_s);
        body->add_return({body_cond, body->add_instruction(migraphx::make_op("neg"), body_x)});
        auto r = mm->add_instruction(loop, {iter, cond, x}, {body});
        mm->add_return({r});
    }
    run_lowering(*p1.get_main_module());

    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto iter  = mm->add_parameter("iter", iter_s);
        auto cond  = mm->add_parameter("cond", cond_s);
        auto x     = mm->add_parameter("x", x_s);
        auto* body = p2.create_module("Loop_0_body");
        body->add_parameter("iter_num", iter_s);
        auto body_cond = body->add_parameter("cond", cond_s);
        auto body_x    = body->add_parameter("x", x_s);
        body->add_return({body_cond, body->add_instruction(migraphx::make_op("neg"), body_x)});
        auto loaded_iter = mm->add_instruction(migraphx::make_op("hip::load_scalar"), iter);
        auto loaded_cond = mm->add_instruction(migraphx::make_op("hip::load_scalar"), cond);
        auto iter_alloc  = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(iter_s)}}));
        auto cond_alloc = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(cond_s)}}));
        auto x_alloc = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(x_s)}}));
        auto output = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(out_s)}}));
        auto cond_out = mm->add_instruction(
            migraphx::make_op("allocate", {{"shape", migraphx::to_value(cond_s)}}));
        auto r = mm->add_instruction(
            migraphx::make_op("gpu::loop", loop.to_value()),
            {loaded_iter, loaded_cond, x, iter_alloc, cond_alloc, x_alloc, cond_out, output},
            {body});
        mm->add_return({r});
    }
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
