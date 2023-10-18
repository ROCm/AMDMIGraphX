/*
 * The MIT License (MIT)
 *
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
#include <iostream>
#include <set>
#include <unordered_set>
#include <vector>
#include <random>
#include <cmath>
#include <migraphx/program.hpp>
#include <migraphx/target.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/module.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/generate_root_modules.hpp>
#include <migraphx/target_assignments.hpp>
#include <test.hpp>

TEST_CASE(fork_case)
{
    /*
            Add (tid = 0)
              |
       ----------------
      |               |
    Mul             Identity
    (tid = 0)       (tid = 1)

    */
    auto s = migraphx::shape{migraphx::shape::float_type, {8}};
    migraphx::target_assignments tass;
    migraphx::program p1;
    {
        auto* mm          = p1.get_main_module();
        auto x_param      = mm->add_parameter("x", s);
        auto y_param      = mm->add_parameter("y", s);
        auto z_param      = mm->add_parameter("z", s);
        auto add_ins      = mm->add_instruction(migraphx::make_op("add"), x_param, y_param);
        auto mul_ins      = mm->add_instruction(migraphx::make_op("mul"), add_ins, z_param);
        auto identity_ins = mm->add_instruction(migraphx::make_op("identity"), add_ins);
        mm->add_return({mul_ins, identity_ins});
        tass.insert(tass.begin(), std::make_pair(add_ins, 0));
        tass.insert(tass.begin(), std::make_pair(mul_ins, 0));
        tass.insert(tass.begin(), std::make_pair(identity_ins, 1));
    }

    migraphx::generate_root_modules(p1, tass);
    migraphx::program p2;
    {
        migraphx::module_ref mm = p2.get_main_module();
        auto y_param = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {8}});
        auto x_param = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {8}});

        migraphx::module_ref target_mod_0_0 = p2.create_module("target_mod_0_0");
        auto target_mod_0_0_param_1         = target_mod_0_0->add_parameter(
            "param:1", migraphx::shape{migraphx::shape::float_type, {8}});
        auto target_mod_0_0_param_0 = target_mod_0_0->add_parameter(
            "param:0", migraphx::shape{migraphx::shape::float_type, {8}});
        auto target_mod_0_0_add = target_mod_0_0->add_instruction(
            migraphx::make_op("add"), target_mod_0_0_param_1, target_mod_0_0_param_0);
        target_mod_0_0->add_return({target_mod_0_0_add});

        auto x_2 = mm->add_instruction(migraphx::make_op("run_on_target", {{"target_id", 0}}),
                                       {y_param, x_param},
                                       {target_mod_0_0});
        auto x_3 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_2);

        auto z_param = mm->add_parameter("z", migraphx::shape{migraphx::shape::float_type, {8}});
        migraphx::module_ref target_mod_1_0 = p2.create_module("target_mod_1_0");
        auto target_mod_1_0_param_0         = target_mod_1_0->add_parameter(
            "param:0", migraphx::shape{migraphx::shape::float_type, {8}});
        auto target_mod_1_0_identity =
            target_mod_1_0->add_instruction(migraphx::make_op("identity"), target_mod_1_0_param_0);
        target_mod_1_0->add_return({target_mod_1_0_identity});

        auto x_5 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 1}}), {x_3}, {target_mod_1_0});
        auto x_6 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_5);

        migraphx::module_ref target_mod_0_1 = p2.create_module("target_mod_0_1");
        auto target_mod_0_1_param_1         = target_mod_0_1->add_parameter(
            "param:1", migraphx::shape{migraphx::shape::float_type, {8}});
        auto target_mod_0_1_param_0 = target_mod_0_1->add_parameter(
            "param:0", migraphx::shape{migraphx::shape::float_type, {8}});
        auto target_mod_0_1_mul = target_mod_0_1->add_instruction(
            migraphx::make_op("mul"), target_mod_0_1_param_1, target_mod_0_1_param_0);
        target_mod_0_1->add_return({target_mod_0_1_mul});

        auto x_7 = mm->add_instruction(migraphx::make_op("run_on_target", {{"target_id", 0}}),
                                       {z_param, x_3},
                                       {target_mod_0_1});
        auto x_8 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_7);
        mm->add_return({x_8, x_6});
    }
    EXPECT(p1.sort() == p2.sort());
};

TEST_CASE(merge_case)
{
    /*
            Add             Identity
            (tid = 0)       (tid = 1)
             |               |
             -----------------
                     |
                    Mul (tid = 0)

    */
    migraphx::target_assignments tass;
    auto s = migraphx::shape{migraphx::shape::float_type, {8}};
    migraphx::program p1;
    {
        auto* mm          = p1.get_main_module();
        auto x_param      = mm->add_parameter("x", s);
        auto y_param      = mm->add_parameter("y", s);
        auto z_param      = mm->add_parameter("z", s);
        auto add_ins      = mm->add_instruction(migraphx::make_op("add"), x_param, y_param);
        auto identity_ins = mm->add_instruction(migraphx::make_op("identity"), z_param);
        auto mul_ins      = mm->add_instruction(migraphx::make_op("mul"), add_ins, identity_ins);
        mm->add_return({mul_ins});
        tass.insert(tass.begin(), std::make_pair(add_ins, 0));
        tass.insert(tass.begin(), std::make_pair(mul_ins, 0));
        tass.insert(tass.begin(), std::make_pair(identity_ins, 1));
    }
    migraphx::generate_root_modules(p1, tass);
    migraphx::program p2;
    {
        migraphx::module_ref mm = p2.get_main_module();
        auto z = mm->add_parameter("z", migraphx::shape{migraphx::shape::float_type, {8}});
        auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {8}});
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {8}});

        migraphx::module_ref target_mod_1_0 = p2.create_module("target_mod_1_0");
        auto target_mod_1_0_param_0         = target_mod_1_0->add_parameter(
            "param:0", migraphx::shape{migraphx::shape::float_type, {8}});
        auto x_target_mod_1_0_1 =
            target_mod_1_0->add_instruction(migraphx::make_op("identity"), target_mod_1_0_param_0);
        target_mod_1_0->add_return({x_target_mod_1_0_1});

        auto x_3 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 1}}), {z}, {target_mod_1_0});
        auto x_4 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_3);

        migraphx::module_ref target_mod_0_0 = p2.create_module("target_mod_0_0");
        auto target_mod_0_0_param_1         = target_mod_0_0->add_parameter(
            "param:1", migraphx::shape{migraphx::shape::float_type, {8}});
        auto target_mod_0_0_param_0 = target_mod_0_0->add_parameter(
            "param:0", migraphx::shape{migraphx::shape::float_type, {8}});
        auto x_target_mod_0_0_2 = target_mod_0_0->add_instruction(
            migraphx::make_op("add"), target_mod_0_0_param_1, target_mod_0_0_param_0);
        target_mod_0_0->add_return({x_target_mod_0_0_2});

        auto x_5 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {y, x}, {target_mod_0_0});
        auto x_6 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_5);

        migraphx::module_ref target_mod_0_1 = p2.create_module("target_mod_0_1");
        auto target_mod_0_1_param_1         = target_mod_0_1->add_parameter(
            "param:1", migraphx::shape{migraphx::shape::float_type, {8}});
        auto target_mod_0_1_param_0 = target_mod_0_1->add_parameter(
            "param:0", migraphx::shape{migraphx::shape::float_type, {8}});
        auto x_target_mod_0_1_2 = target_mod_0_1->add_instruction(
            migraphx::make_op("mul"), target_mod_0_1_param_1, target_mod_0_1_param_0);
        target_mod_0_1->add_return({x_target_mod_0_1_2});

        auto x_7 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {x_4, x_6}, {target_mod_0_1});
        auto x_8 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_7);
        mm->add_return({x_8});
    }
    EXPECT(p1.sort() == p2.sort());
};

TEST_CASE(fork_and_merge_case)
{
    /*
           Add (tid = 0)
            |
     ----------------
     |               |
   Mul             Identity
   (tid = 0)       (tid = 1)
     |               |
     ----------------
            |
          Sub (tid = 0)
   */

    auto s = migraphx::shape{migraphx::shape::float_type, {8}};
    migraphx::target_assignments tass;
    migraphx::program p1;
    {
        auto* mm          = p1.get_main_module();
        auto x_param      = mm->add_parameter("x", s);
        auto y_param      = mm->add_parameter("y", s);
        auto z_param      = mm->add_parameter("z", s);
        auto add_ins      = mm->add_instruction(migraphx::make_op("add"), x_param, y_param);
        auto mul_ins      = mm->add_instruction(migraphx::make_op("mul"), add_ins, z_param);
        auto identity_ins = mm->add_instruction(migraphx::make_op("identity"), add_ins);
        auto merge_ins    = mm->add_instruction(migraphx::make_op("sub"), identity_ins, mul_ins);
        tass.insert(tass.begin(), std::make_pair(add_ins, 0));
        tass.insert(tass.begin(), std::make_pair(mul_ins, 0));
        tass.insert(tass.begin(), std::make_pair(identity_ins, 1));
        tass.insert(tass.begin(), std::make_pair(merge_ins, 0));
        mm->add_return({merge_ins});
    }
    migraphx::generate_root_modules(p1, tass);
    migraphx::program p2;
    {
        migraphx::module_ref mm = p2.get_main_module();
        auto z = mm->add_parameter("z", migraphx::shape{migraphx::shape::float_type, {8}});
        auto y = mm->add_parameter("y", migraphx::shape{migraphx::shape::float_type, {8}});
        auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {8}});

        migraphx::module_ref target_mod_0_0 = p2.create_module("target_mod_0_0");
        auto target_mod_0_0_param_1         = target_mod_0_0->add_parameter(
            "param:1", migraphx::shape{migraphx::shape::float_type, {8}});
        auto target_mod_0_0_param_0 = target_mod_0_0->add_parameter(
            "param:0", migraphx::shape{migraphx::shape::float_type, {8}});
        auto x_target_mod_0_0_2 = target_mod_0_0->add_instruction(
            migraphx::make_op("add"), target_mod_0_0_param_1, target_mod_0_0_param_0);
        target_mod_0_0->add_return({x_target_mod_0_0_2});

        auto x_3 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {y, x}, {target_mod_0_0});
        auto x_4 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_3);

        migraphx::module_ref target_mod_0_1 = p2.create_module("target_mod_0_1");
        auto target_mod_0_1_param_1         = target_mod_0_1->add_parameter(
            "param:1", migraphx::shape{migraphx::shape::float_type, {8}});
        auto target_mod_0_1_param_0 = target_mod_0_1->add_parameter(
            "param:0", migraphx::shape{migraphx::shape::float_type, {8}});
        auto x_target_mod_0_1_2 = target_mod_0_1->add_instruction(
            migraphx::make_op("mul"), target_mod_0_1_param_1, target_mod_0_1_param_0);
        target_mod_0_1->add_return({x_target_mod_0_1_2});

        auto x_5 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {z, x_4}, {target_mod_0_1});
        auto x_6 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_5);

        migraphx::module_ref target_mod_1_0 = p2.create_module("target_mod_1_0");
        auto target_mod_1_0_param_0         = target_mod_1_0->add_parameter(
            "param:0", migraphx::shape{migraphx::shape::float_type, {8}});
        auto x_target_mod_1_0_1 =
            target_mod_1_0->add_instruction(migraphx::make_op("identity"), target_mod_1_0_param_0);
        target_mod_1_0->add_return({x_target_mod_1_0_1});

        auto x_7 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 1}}), {x_4}, {target_mod_1_0});
        auto x_8 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_7);

        migraphx::module_ref target_mod_0_2 = p2.create_module("target_mod_0_2");
        auto target_mod_0_2_param_1         = target_mod_0_2->add_parameter(
            "param:1", migraphx::shape{migraphx::shape::float_type, {8}});
        auto target_mod_0_2_param_0 = target_mod_0_2->add_parameter(
            "param:0", migraphx::shape{migraphx::shape::float_type, {8}});
        auto x_target_mod_0_2_2 = target_mod_0_2->add_instruction(
            migraphx::make_op("sub"), target_mod_0_2_param_1, target_mod_0_2_param_0);
        target_mod_0_2->add_return({x_target_mod_0_2_2});

        auto x_9 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {x_6, x_8}, {target_mod_0_2});
        auto x_10 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_9);
        mm->add_return({x_10});
    }
    p1.print_cpp(std::cout);
    EXPECT(p1.sort() == p2.sort());
};

int main(int argc, const char* argv[]) { test::run(argc, argv); }
