/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>
#include <pointwise.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::fuse_pointwise{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(single)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto z    = mm->add_parameter("z", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), pass, z);
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto z    = mm->add_parameter("z", s);
        auto add1 = add_pointwise(p2, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = add_pointwise(p2, "main:pointwise1", {pass, z}, single_pointwise("add"));
        mm->add_return({add2});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(double_add)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto z    = mm->add_parameter("z", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, z);
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto z   = mm->add_parameter("z", s);
        auto fadd =
            add_pointwise(p2, "main:pointwise0", {x, y, z}, [=](auto* pm, const auto& inputs) {
                auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("add"), add1, inputs[2]);
            });
        mm->add_return({fadd});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(double_add_without_return)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto z    = mm->add_parameter("z", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        mm->add_instruction(migraphx::make_op("add"), add1, z);
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto z   = mm->add_parameter("z", s);
        auto fadd =
            add_pointwise(p2, "main:pointwise0", {x, y, z}, [=](auto* pm, const auto& inputs) {
                auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("add"), add1, inputs[2]);
            });
        mm->add_instruction(migraphx::make_op("identity"), fadd);
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(used_twice_not_fused)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, y);
        auto add3 = mm->add_instruction(migraphx::make_op("add"), pass, add2);
        mm->add_return({add3});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = add_pointwise(p2, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto fadd = add_pointwise(
            p2, "main:pointwise1", {add1, y, pass}, [=](auto* pm, const auto& inputs) {
                auto add2 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("add"), inputs[2], add2);
            });
        mm->add_return({fadd});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(used_twice_fused)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), add1, x);
        auto add3 = mm->add_instruction(migraphx::make_op("add"), add1, y);
        auto add4 = mm->add_instruction(migraphx::make_op("add"), add2, add3);
        mm->add_return({add4});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto fadd = add_pointwise(p2, "main:pointwise0", {x, y}, [=](auto* pm, const auto& inputs) {
            auto add1 = pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
            auto add2 = pm->add_instruction(migraphx::make_op("add"), add1, inputs[0]);
            auto add3 = pm->add_instruction(migraphx::make_op("add"), add1, inputs[1]);
            return pm->add_instruction(migraphx::make_op("add"), add2, add3);
        });
        mm->add_return({fadd});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(duplicate_inputs)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, x);
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), pass, y);
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = add_pointwise(p2, "main:pointwise0", {x}, [=](auto* pm, const auto& inputs) {
            return pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[0]);
        });
        auto pass = mm->add_instruction(pass_op{}, add1);
        auto add2 = add_pointwise(p2, "main:pointwise1", {pass, y}, single_pointwise("add"));
        mm->add_return({add2});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(scalar_input)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto one = mm->add_literal(1.0f);
        auto y =
            mm->add_instruction(migraphx::make_op("scalar", {{"scalar_bcst_dims", s.lens()}}), one);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        mm->add_return({add1});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto add1 = add_pointwise(p2, "main:pointwise0", {x}, [=](auto* pm, const auto& inputs) {
            auto y = pm->add_literal(1.0f);
            return pm->add_instruction(migraphx::make_op("add"), inputs[0], y);
        });
        mm->add_return({add1});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(contiguous_input)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto one = mm->add_literal(1.0f);
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), one);
        auto y    = mm->add_instruction(migraphx::make_op("contiguous"), yb);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        mm->add_return({add1});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto add1 = add_pointwise(p2, "main:pointwise0", {x}, [=](auto* pm, const auto& inputs) {
            auto y = pm->add_literal(1.0f);
            return pm->add_instruction(migraphx::make_op("add"), inputs[0], y);
        });
        mm->add_return({add1});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(contiguous_boolean_input)
{

    migraphx::shape s{migraphx::shape::bool_type, {2, 3}};
    migraphx::shape s_lit{migraphx::shape::bool_type, {1}, {0}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto one = mm->add_literal(migraphx::literal(s_lit, {1.0}));
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), one);
        auto y    = mm->add_instruction(migraphx::make_op("contiguous"), yb);
        auto xor1 = mm->add_instruction(migraphx::make_op("logical_xor"), x, y);
        mm->add_return({xor1});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto xor1 = add_pointwise(p2, "main:pointwise0", {x}, [=](auto* pm, const auto& inputs) {
            auto y = pm->add_literal(migraphx::literal(s_lit, {1}));
            return pm->add_instruction(migraphx::make_op("logical_xor"), inputs[0], y);
        });
        mm->add_return({xor1});
    }
}

TEST_CASE(all_scalar_input)
{
    migraphx::shape s{migraphx::shape::float_type};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), x, y);
        mm->add_return({add1});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add1 = add_pointwise(p2, "main:pointwise0", {x, y}, [=](auto* pm, const auto& inputs) {
            return pm->add_instruction(migraphx::make_op("add"), inputs[0], inputs[1]);
        });
        mm->add_return({add1});
    }
    EXPECT(p1.get_output_shapes().size() == 1);
    EXPECT(p1.get_output_shapes().front().scalar());
    EXPECT(p1.get_output_shapes() == p2.get_output_shapes());
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
