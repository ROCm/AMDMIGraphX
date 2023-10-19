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

TEST_CASE(single_target_test)
{
    /*
        Add (tid = 1)
         |
        Return
    */
    migraphx::target_assignments tass;
    auto s = migraphx::shape{migraphx::shape::float_type, {8}};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto x_param = mm->add_parameter("x", s);
        auto y_param = mm->add_parameter("y", s);
        auto add_ins = mm->add_instruction(migraphx::make_op("add"), x_param, y_param);
        mm->add_return({add_ins});
        tass.insert(tass.begin(), std::make_pair(add_ins, 1));
    }
    migraphx::generate_root_modules(p1, tass);
    migraphx::program p2;
    {
        migraphx::module_ref mm             = p2.get_main_module();
        auto y                              = mm->add_parameter("y", s);
        auto x                              = mm->add_parameter("x", s);
        migraphx::module_ref target_mod_1_0 = p2.create_module("target_mod_1_0");
        auto target_mod_1_0_param_1         = target_mod_1_0->add_parameter("param:1", s);
        auto target_mod_1_0_param_0         = target_mod_1_0->add_parameter("param:0", s);
        auto x_target_mod_1_0_2             = target_mod_1_0->add_instruction(
            migraphx::make_op("add"), target_mod_1_0_param_1, target_mod_1_0_param_0);
        target_mod_1_0->add_return({x_target_mod_1_0_2});

        auto x_2 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 1}}), {y, x}, {target_mod_1_0});
        auto x_3 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_2);
        mm->add_return({x_3});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(two_targets_with_ref)
{
    /*
        Identity
         |
        Add (tid = 1)
         |
        Mul (tid = 0)
         |
        Identity
         |
        Return
    */
    migraphx::target_assignments tass;
    auto s = migraphx::shape{migraphx::shape::float_type, {8}};
    migraphx::program p1;
    {
        auto* mm            = p1.get_main_module();
        auto x_param        = mm->add_parameter("x", s);
        auto y_param        = mm->add_parameter("y", s);
        auto z_param        = mm->add_parameter("z", s);
        auto identity_ins_0 = mm->add_instruction(migraphx::make_op("identity"), x_param);
        auto add_ins = mm->add_instruction(migraphx::make_op("add"), identity_ins_0, y_param);
        auto mul_ins = mm->add_instruction(migraphx::make_op("mul"), add_ins, z_param);
        auto identity_ins_1 = mm->add_instruction(migraphx::make_op("identity"), mul_ins);
        mm->add_return({identity_ins_1});
        tass.insert(tass.begin(), std::make_pair(add_ins, 1));
        tass.insert(tass.begin(), std::make_pair(mul_ins, 0));
    }
    migraphx::generate_root_modules(p1, tass);
    migraphx::program p2;
    {
        migraphx::module_ref mm             = p2.get_main_module();
        auto z                              = mm->add_parameter("z", s);
        auto y                              = mm->add_parameter("y", s);
        auto x                              = mm->add_parameter("x", s);
        auto identity_ins_0                 = mm->add_instruction(migraphx::make_op("identity"), x);
        migraphx::module_ref target_mod_1_0 = p2.create_module("target_mod_1_0");
        auto target_mod_1_0_param_1         = target_mod_1_0->add_parameter(
            "param:1", migraphx::shape{migraphx::shape::float_type, {8}});
        auto target_mod_1_0_param_0 = target_mod_1_0->add_parameter(
            "param:0", migraphx::shape{migraphx::shape::float_type, {8}});
        auto x_target_mod_1_0_2 = target_mod_1_0->add_instruction(
            migraphx::make_op("add"), target_mod_1_0_param_1, target_mod_1_0_param_0);
        target_mod_1_0->add_return({x_target_mod_1_0_2});

        migraphx::module_ref target_mod_0_0 = p2.create_module("target_mod_0_0");
        auto target_mod_0_0_param_1         = target_mod_0_0->add_parameter(
            "param:1", migraphx::shape{migraphx::shape::float_type, {8}});
        auto target_mod_0_0_param_0 = target_mod_0_0->add_parameter(
            "param:0", migraphx::shape{migraphx::shape::float_type, {8}});
        auto x_target_mod_0_0_2 = target_mod_0_0->add_instruction(
            migraphx::make_op("mul"), target_mod_0_0_param_1, target_mod_0_0_param_0);
        target_mod_0_0->add_return({x_target_mod_0_0_2});

        auto x_3 = mm->add_instruction(migraphx::make_op("run_on_target", {{"target_id", 1}}),
                                       {y, identity_ins_0},
                                       {target_mod_1_0});
        auto x_4 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_3);

        auto x_5 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {z, x_4}, {target_mod_0_0});
        auto x_6 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_5);
        auto identity_ins_1 = mm->add_instruction(migraphx::make_op("identity"), x_6);
        mm->add_return({identity_ins_1});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(two_targets_ref_inbetween)
{
    /*
        Identity
         |
        Add (tid = 1)
         |
        Identity
         |
        Mul (tid = 0)
         |
        Return
    */
    migraphx::target_assignments tass;
    auto s = migraphx::shape{migraphx::shape::float_type, {8}};
    migraphx::program p1;
    {
        auto* mm            = p1.get_main_module();
        auto x_param        = mm->add_parameter("x", s);
        auto y_param        = mm->add_parameter("y", s);
        auto z_param        = mm->add_parameter("z", s);
        auto identity_ins_0 = mm->add_instruction(migraphx::make_op("identity"), x_param);
        auto add_ins = mm->add_instruction(migraphx::make_op("add"), identity_ins_0, y_param);
        auto identity_ins_1 = mm->add_instruction(migraphx::make_op("identity"), add_ins);
        auto mul_ins = mm->add_instruction(migraphx::make_op("mul"), identity_ins_1, z_param);
        mm->add_return({mul_ins});
        tass.insert(tass.begin(), std::make_pair(add_ins, 1));
        tass.insert(tass.begin(), std::make_pair(mul_ins, 0));
    }
    migraphx::generate_root_modules(p1, tass);
    migraphx::program p2;
    {
        migraphx::module_ref mm = p2.get_main_module();
        auto z                  = mm->add_parameter("z", s);
        auto y                  = mm->add_parameter("y", s);
        auto x                  = mm->add_parameter("x", s);
        auto identity_ins       = mm->add_instruction(migraphx::make_op("identity"), x);

        migraphx::module_ref target_mod_1_0 = p2.create_module("target_mod_1_0");
        auto target_mod_1_0_param_1         = target_mod_1_0->add_parameter(
            "param:1", migraphx::shape{migraphx::shape::float_type, {8}});
        auto target_mod_1_0_param_0 = target_mod_1_0->add_parameter(
            "param:0", migraphx::shape{migraphx::shape::float_type, {8}});
        auto x_target_mod_1_0_2 = target_mod_1_0->add_instruction(
            migraphx::make_op("add"), target_mod_1_0_param_1, target_mod_1_0_param_0);
        target_mod_1_0->add_return({x_target_mod_1_0_2});

        migraphx::module_ref target_mod_0_0 = p2.create_module("target_mod_0_0");
        auto target_mod_0_0_param_1         = target_mod_0_0->add_parameter(
            "param:1", migraphx::shape{migraphx::shape::float_type, {8}});
        auto target_mod_0_0_param_0 = target_mod_0_0->add_parameter(
            "param:0", migraphx::shape{migraphx::shape::float_type, {8}});
        auto x_target_mod_0_0_2 = target_mod_0_0->add_instruction(
            migraphx::make_op("mul"), target_mod_0_0_param_1, target_mod_0_0_param_0);
        target_mod_0_0->add_return({x_target_mod_0_0_2});

        auto x_3 = mm->add_instruction(migraphx::make_op("run_on_target", {{"target_id", 1}}),
                                       {y, identity_ins},
                                       {target_mod_1_0});
        auto x_4 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_3);
        auto x_5 = mm->add_instruction(migraphx::make_op("identity"), x_4);

        auto x_6 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {z, x_5}, {target_mod_0_0});
        auto x_7 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_6);
        mm->add_return({x_7});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(if_then_else_program)
{
    /*
                If -----------------> Return
                |
          ---------------
          |             |
       (then_mod)      (else_mod)
          |             |
        Add (tid = 0)  Mul (tid = 1)
    */

    migraphx::target_assignments tass;
    migraphx::shape cond_s{migraphx::shape::bool_type};
    migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data1(ds.elements(), 1);
    std::vector<float> data2(ds.elements(), 2);
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto cond = mm->add_parameter("cond", cond_s);
        auto x    = mm->add_parameter("x", ds);
        auto y    = mm->add_parameter("y", ds);

        auto* then_mod = p1.create_module("if_gpu_mod");
        auto l1        = then_mod->add_literal(migraphx::literal(ds, data1));
        auto a1        = then_mod->add_instruction(migraphx::make_op("add"), x, l1);
        then_mod->add_return({a1});

        auto* else_mod = p1.create_module("else_cpu_mod");
        auto l2        = else_mod->add_literal(migraphx::literal(ds, data2));
        auto a2        = else_mod->add_instruction(migraphx::make_op("mul"), y, l2);
        else_mod->add_return({a2});

        auto ret = mm->add_instruction(migraphx::make_op("if"), {cond}, {then_mod, else_mod});
        auto r   = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
        mm->add_return({r});
        tass.insert(tass.begin(), std::make_pair(l1, 0));
        tass.insert(tass.begin(), std::make_pair(a1, 0));
        tass.insert(tass.begin(), std::make_pair(l2, 1));
        tass.insert(tass.begin(), std::make_pair(a2, 1));
    }
    migraphx::generate_root_modules(p1, tass);
    migraphx::program p2;
    {
        migraphx::module_ref mm = p2.get_main_module();
        auto x                  = mm->add_parameter("x", ds);
        auto y                  = mm->add_parameter("y", ds);
        auto cond               = mm->add_parameter("cond", cond_s);

        migraphx::module_ref target_mod_0_0 = p2.create_module("target_mod_0_0");
        auto target_mod_0_0_param_1         = target_mod_0_0->add_parameter(
            "param:1", migraphx::shape{migraphx::shape::float_type, {2, 3}});
        auto target_mod_0_0_param_0 = target_mod_0_0->add_parameter(
            "param:0", migraphx::shape{migraphx::shape::float_type, {2, 3}});
        auto x_target_mod_0_0_2 = target_mod_0_0->add_instruction(
            migraphx::make_op("add"), target_mod_0_0_param_1, target_mod_0_0_param_0);
        target_mod_0_0->add_return({x_target_mod_0_0_2});

        migraphx::module_ref if_gpu_mod = p2.create_module("if_gpu_mod");
        auto x_if_gpu_mod_0             = if_gpu_mod->add_literal(migraphx::literal(ds, data1));
        auto x_if_gpu_mod_1 =
            if_gpu_mod->add_instruction(migraphx::make_op("run_on_target", {{"target_id", 0}}),
                                        {x_if_gpu_mod_0, x},
                                        {target_mod_0_0});
        auto x_if_gpu_mod_2 = if_gpu_mod->add_instruction(
            migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_if_gpu_mod_1);
        if_gpu_mod->add_return({x_if_gpu_mod_2});

        migraphx::module_ref target_mod_1_0 = p2.create_module("target_mod_1_0");
        auto target_mod_1_0_param_1         = target_mod_1_0->add_parameter(
            "param:1", migraphx::shape{migraphx::shape::float_type, {2, 3}});
        auto target_mod_1_0_param_0 = target_mod_1_0->add_parameter(
            "param:0", migraphx::shape{migraphx::shape::float_type, {2, 3}});
        auto x_target_mod_1_0_2 = target_mod_1_0->add_instruction(
            migraphx::make_op("mul"), target_mod_1_0_param_1, target_mod_1_0_param_0);
        target_mod_1_0->add_return({x_target_mod_1_0_2});

        migraphx::module_ref else_cpu_mod = p2.create_module("else_cpu_mod");
        auto x_else_cpu_mod_0             = else_cpu_mod->add_literal(migraphx::literal(ds, data2));
        auto x_else_cpu_mod_1 =
            else_cpu_mod->add_instruction(migraphx::make_op("run_on_target", {{"target_id", 1}}),
                                          {x_else_cpu_mod_0, y},
                                          {target_mod_1_0});
        auto x_else_cpu_mod_2 = else_cpu_mod->add_instruction(
            migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_else_cpu_mod_1);
        else_cpu_mod->add_return({x_else_cpu_mod_2});

        auto x_3 = mm->add_instruction(migraphx::make_op("if"), {cond}, {if_gpu_mod, else_cpu_mod});
        auto x_4 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_3);
        mm->add_return({x_4});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(fork_case_1)
{
    /*
            Add (tid = 0)
              |
       ---------------
      |               |
    Mul             Identity
    (tid = 0)       (tid = 1)
      |               |
      Return         Return

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
        auto y_param            = mm->add_parameter("y", s);
        auto x_param            = mm->add_parameter("x", s);

        migraphx::module_ref target_mod_0_0 = p2.create_module("target_mod_0_0");
        auto target_mod_0_0_param_1         = target_mod_0_0->add_parameter("param:1", s);
        auto target_mod_0_0_param_0         = target_mod_0_0->add_parameter("param:0", s);
        auto target_mod_0_0_add             = target_mod_0_0->add_instruction(
            migraphx::make_op("add"), target_mod_0_0_param_1, target_mod_0_0_param_0);
        target_mod_0_0->add_return({target_mod_0_0_add});

        auto x_2 = mm->add_instruction(migraphx::make_op("run_on_target", {{"target_id", 0}}),
                                       {y_param, x_param},
                                       {target_mod_0_0});
        auto x_3 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_2);

        auto z_param                        = mm->add_parameter("z", s);
        migraphx::module_ref target_mod_1_0 = p2.create_module("target_mod_1_0");
        auto target_mod_1_0_param_0         = target_mod_1_0->add_parameter("param:0", s);
        auto target_mod_1_0_identity =
            target_mod_1_0->add_instruction(migraphx::make_op("identity"), target_mod_1_0_param_0);
        target_mod_1_0->add_return({target_mod_1_0_identity});

        auto x_5 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 1}}), {x_3}, {target_mod_1_0});
        auto x_6 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_5);

        migraphx::module_ref target_mod_0_1 = p2.create_module("target_mod_0_1");
        auto target_mod_0_1_param_1         = target_mod_0_1->add_parameter("param:1", s);
        auto target_mod_0_1_param_0         = target_mod_0_1->add_parameter("param:0", s);
        auto target_mod_0_1_mul             = target_mod_0_1->add_instruction(
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

TEST_CASE(fork_case_2)
{
    /*
                Add (no assignment)
                  |
           ---------------
          |               |
        Mul             Identity
        (no assignment) (no assignment)
          |               |
          Return         Return

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
    }
    migraphx::program p2 = p1;
    migraphx::generate_root_modules(p1, tass);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(fork_case_3)
{
    /*
                Add (no assignment)
                  |
           ---------------
          |               |
        Mul             Identity
        (tid = 0)    (no assignment)
          |               |
          Return         Return

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
        tass.insert(tass.begin(), std::make_pair(mul_ins, 0));
    }
    migraphx::generate_root_modules(p1, tass);
    migraphx::program p2;
    {
        migraphx::module_ref mm = p2.get_main_module();
        auto y                  = mm->add_parameter("y", s);
        auto x                  = mm->add_parameter("x", s);
        auto x_2                = mm->add_instruction(migraphx::make_op("add"), x, y);
        auto x_3                = mm->add_instruction(migraphx::make_op("identity"), x_2);
        auto z                  = mm->add_parameter("z", s);

        migraphx::module_ref target_mod_0_0 = p2.create_module("target_mod_0_0");
        auto target_mod_0_0_param_1         = target_mod_0_0->add_parameter("param:1", s);
        auto target_mod_0_0_param_0         = target_mod_0_0->add_parameter("param:0", s);
        auto x_target_mod_0_0_2             = target_mod_0_0->add_instruction(
            migraphx::make_op("mul"), target_mod_0_0_param_1, target_mod_0_0_param_0);
        target_mod_0_0->add_return({x_target_mod_0_0_2});

        auto x_5 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {z, x_2}, {target_mod_0_0});
        auto x_6 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_5);
        mm->add_return({x_6, x_3});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(merge_case_1)
{
    /*
            Add             Identity
            (tid = 0)       (tid = 1)
             |               |
             -----------------
                     |
                    Mul (tid = 0)
                     |
                   Return

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
        auto z                  = mm->add_parameter("z", s);
        auto y                  = mm->add_parameter("y", s);
        auto x                  = mm->add_parameter("x", s);

        migraphx::module_ref target_mod_1_0 = p2.create_module("target_mod_1_0");
        auto target_mod_1_0_param_0         = target_mod_1_0->add_parameter("param:0", s);
        auto x_target_mod_1_0_1 =
            target_mod_1_0->add_instruction(migraphx::make_op("identity"), target_mod_1_0_param_0);
        target_mod_1_0->add_return({x_target_mod_1_0_1});

        auto x_3 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 1}}), {z}, {target_mod_1_0});
        auto x_4 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_3);

        migraphx::module_ref target_mod_0_0 = p2.create_module("target_mod_0_0");
        auto target_mod_0_0_param_1         = target_mod_0_0->add_parameter("param:1", s);
        auto target_mod_0_0_param_0         = target_mod_0_0->add_parameter("param:0", s);
        auto x_target_mod_0_0_2             = target_mod_0_0->add_instruction(
            migraphx::make_op("add"), target_mod_0_0_param_1, target_mod_0_0_param_0);
        target_mod_0_0->add_return({x_target_mod_0_0_2});

        auto x_5 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {y, x}, {target_mod_0_0});
        auto x_6 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_5);

        migraphx::module_ref target_mod_0_1 = p2.create_module("target_mod_0_1");
        auto target_mod_0_1_param_1         = target_mod_0_1->add_parameter("param:1", s);
        auto target_mod_0_1_param_0         = target_mod_0_1->add_parameter("param:0", s);
        auto x_target_mod_0_1_2             = target_mod_0_1->add_instruction(
            migraphx::make_op("mul"), target_mod_0_1_param_1, target_mod_0_1_param_0);
        target_mod_0_1->add_return({x_target_mod_0_1_2});

        auto x_7 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {x_4, x_6}, {target_mod_0_1});
        auto x_8 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_7);
        mm->add_return({x_8});
    }
    EXPECT(p1.sort() == p2.sort());
};

TEST_CASE(merge_case_2)
{
    /*
                Add             Identity
                (no assignment) (no assignment)
                 |               |
                 -----------------
                         |
                        Mul (no assignment)
                         |
                       Return

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
    }
    migraphx::program p2 = p1;
    migraphx::generate_root_modules(p1, tass);
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(merge_case_3)
{
    /*
                Add             Identity
                (tid=0)       (no assignment)
                 |               |
                 -----------------
                         |
                        Mul (no assignment)
                         |
                       Return

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
    }
    migraphx::generate_root_modules(p1, tass);
    migraphx::program p2;
    {
        migraphx::module_ref mm = p2.get_main_module();
        auto z                  = mm->add_parameter("z", s);
        auto x_1                = mm->add_instruction(migraphx::make_op("identity"), z);
        auto y                  = mm->add_parameter("y", s);
        auto x                  = mm->add_parameter("x", s);

        migraphx::module_ref target_mod_0_0 = p2.create_module("target_mod_0_0");
        auto target_mod_0_0_param_1         = target_mod_0_0->add_parameter("param:1", s);
        auto target_mod_0_0_param_0         = target_mod_0_0->add_parameter("param:0", s);
        auto x_target_mod_0_0_2             = target_mod_0_0->add_instruction(
            migraphx::make_op("add"), target_mod_0_0_param_1, target_mod_0_0_param_0);
        target_mod_0_0->add_return({x_target_mod_0_0_2});

        auto x_4 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {y, x}, {target_mod_0_0});
        auto x_5 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_4);
        auto x_6 = mm->add_instruction(migraphx::make_op("mul"), x_5, x_1);
        mm->add_return({x_6});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(merge_case_4)
{
    /*
     **** "Return" as the Merge Node ****
                Add             Identity
                (tid=0)       (no assignment)
                 |               |
                 -----------------
                         |
                        Return
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
        mm->add_return({add_ins, identity_ins});
        tass.insert(tass.begin(), std::make_pair(add_ins, 0));
    }
    migraphx::generate_root_modules(p1, tass);
    migraphx::program p2;
    {
        migraphx::module_ref mm = p2.get_main_module();
        auto z                  = mm->add_parameter("z", s);
        auto x_1                = mm->add_instruction(migraphx::make_op("identity"), z);
        auto y                  = mm->add_parameter("y", s);
        auto x                  = mm->add_parameter("x", s);

        migraphx::module_ref target_mod_0_0 = p2.create_module("target_mod_0_0");
        auto target_mod_0_0_param_1         = target_mod_0_0->add_parameter("param:1", s);
        auto target_mod_0_0_param_0         = target_mod_0_0->add_parameter("param:0", s);
        auto x_target_mod_0_0_2             = target_mod_0_0->add_instruction(
            migraphx::make_op("add"), target_mod_0_0_param_1, target_mod_0_0_param_0);
        target_mod_0_0->add_return({x_target_mod_0_0_2});

        auto x_4 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {y, x}, {target_mod_0_0});
        auto x_5 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_4);
        mm->add_return({x_5, x_1});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(merge_case_5)
{
    /*
         **** "Return" as the Merge Node ****
                 Add (tid = 0)
                     |
                 Identity             Identity
            (no assignment)       (no assignment)
                     |               |
                     -----------------
                             |
                            Return
    */
    migraphx::target_assignments tass;
    auto s = migraphx::shape{migraphx::shape::float_type, {8}};
    migraphx::program p1;
    {
        auto* mm            = p1.get_main_module();
        auto x_param        = mm->add_parameter("x", s);
        auto y_param        = mm->add_parameter("y", s);
        auto z_param        = mm->add_parameter("z", s);
        auto add_ins        = mm->add_instruction(migraphx::make_op("add"), x_param, y_param);
        auto identity_ins_0 = mm->add_instruction(migraphx::make_op("identity"), add_ins);
        auto identity_ins_1 = mm->add_instruction(migraphx::make_op("identity"), z_param);
        mm->add_return({identity_ins_0, identity_ins_1});
        tass.insert(tass.begin(), std::make_pair(add_ins, 0));
    }
    migraphx::generate_root_modules(p1, tass);
    migraphx::program p2;
    {
        migraphx::module_ref mm = p2.get_main_module();
        auto z                  = mm->add_parameter("z", s);
        auto x_1                = mm->add_instruction(migraphx::make_op("identity"), z);
        auto y                  = mm->add_parameter("y", s);
        auto x                  = mm->add_parameter("x", s);

        migraphx::module_ref target_mod_0_0 = p2.create_module("target_mod_0_0");
        auto target_mod_0_0_param_1         = target_mod_0_0->add_parameter("param:1", s);
        auto target_mod_0_0_param_0         = target_mod_0_0->add_parameter("param:0", s);
        auto x_target_mod_0_0_2             = target_mod_0_0->add_instruction(
            migraphx::make_op("add"), target_mod_0_0_param_1, target_mod_0_0_param_0);
        target_mod_0_0->add_return({x_target_mod_0_0_2});

        auto x_4 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {y, x}, {target_mod_0_0});
        auto x_5 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_4);
        auto x_6 = mm->add_instruction(migraphx::make_op("identity"), x_5);
        mm->add_return({x_6, x_1});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(fork_and_merge_case_1)
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
            |
         Return
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
        auto z                  = mm->add_parameter("z", s);
        auto y                  = mm->add_parameter("y", s);
        auto x                  = mm->add_parameter("x", s);

        migraphx::module_ref target_mod_0_0 = p2.create_module("target_mod_0_0");
        auto target_mod_0_0_param_1         = target_mod_0_0->add_parameter("param:1", s);
        auto target_mod_0_0_param_0         = target_mod_0_0->add_parameter("param:0", s);
        auto x_target_mod_0_0_2             = target_mod_0_0->add_instruction(
            migraphx::make_op("add"), target_mod_0_0_param_1, target_mod_0_0_param_0);
        target_mod_0_0->add_return({x_target_mod_0_0_2});

        auto x_3 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {y, x}, {target_mod_0_0});
        auto x_4 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_3);

        migraphx::module_ref target_mod_0_1 = p2.create_module("target_mod_0_1");
        auto target_mod_0_1_param_1         = target_mod_0_1->add_parameter("param:1", s);
        auto target_mod_0_1_param_0         = target_mod_0_1->add_parameter("param:0", s);
        auto x_target_mod_0_1_2             = target_mod_0_1->add_instruction(
            migraphx::make_op("mul"), target_mod_0_1_param_1, target_mod_0_1_param_0);
        target_mod_0_1->add_return({x_target_mod_0_1_2});

        auto x_5 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {z, x_4}, {target_mod_0_1});
        auto x_6 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_5);

        migraphx::module_ref target_mod_1_0 = p2.create_module("target_mod_1_0");
        auto target_mod_1_0_param_0         = target_mod_1_0->add_parameter("param:0", s);
        auto x_target_mod_1_0_1 =
            target_mod_1_0->add_instruction(migraphx::make_op("identity"), target_mod_1_0_param_0);
        target_mod_1_0->add_return({x_target_mod_1_0_1});

        auto x_7 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 1}}), {x_4}, {target_mod_1_0});
        auto x_8 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_7);

        migraphx::module_ref target_mod_0_2 = p2.create_module("target_mod_0_2");
        auto target_mod_0_2_param_1         = target_mod_0_2->add_parameter("param:1", s);
        auto target_mod_0_2_param_0         = target_mod_0_2->add_parameter("param:0", s);
        auto x_target_mod_0_2_2             = target_mod_0_2->add_instruction(
            migraphx::make_op("sub"), target_mod_0_2_param_1, target_mod_0_2_param_0);
        target_mod_0_2->add_return({x_target_mod_0_2_2});

        auto x_9 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {x_6, x_8}, {target_mod_0_2});
        auto x_10 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_9);
        mm->add_return({x_10});
    }
    EXPECT(p1.sort() == p2.sort());
};

TEST_CASE(fork_and_merge_case_2)
{
    /*
        **** Fork node returning ****

                   Add (tid = 0)
                        |
            ---------------------------
            |                         |
        Identity (tid  = 0)           |
            |                         |
            --------------------------
                        |
                     Return
    */
    auto s = migraphx::shape{migraphx::shape::float_type, {8}};
    migraphx::target_assignments tass;
    migraphx::program p1;
    {
        auto* mm          = p1.get_main_module();
        auto x_param      = mm->add_parameter("x", s);
        auto y_param      = mm->add_parameter("y", s);
        auto add_ins      = mm->add_instruction(migraphx::make_op("add"), x_param, y_param);
        auto identity_ins = mm->add_instruction(migraphx::make_op("identity"), add_ins);
        mm->add_return({add_ins, identity_ins});
        tass.insert(tass.begin(), std::make_pair(add_ins, 0));
        tass.insert(tass.begin(), std::make_pair(identity_ins, 0));
    }
    migraphx::generate_root_modules(p1, tass);
    migraphx::program p2;
    {
        migraphx::module_ref mm = p2.get_main_module();
        auto y                  = mm->add_parameter("y", s);
        auto x                  = mm->add_parameter("x", s);

        migraphx::module_ref target_mod_0_0 = p2.create_module("target_mod_0_0");
        auto target_mod_0_0_param_0         = target_mod_0_0->add_parameter("param:0", s);
        auto target_mod_0_0_param_1         = target_mod_0_0->add_parameter("param:1", s);
        auto x_target_mod_0_0_0             = target_mod_0_0->add_instruction(
            migraphx::make_op("add"), target_mod_0_0_param_1, target_mod_0_0_param_0);
        auto x_target_mod_0_0_1 =
            target_mod_0_0->add_instruction(migraphx::make_op("identity"), x_target_mod_0_0_0);
        target_mod_0_0->add_return({x_target_mod_0_0_0, x_target_mod_0_0_1});

        auto x_2 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {y, x}, {target_mod_0_0});
        auto x_3 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_2);
        auto x_4 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), x_2);
        mm->add_return({x_3, x_4});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(fork_and_merge_case_3)
{
    /*
        **** Fork node returning ****

                  Add (tid = 0)
                   |
                Identity (no target_assignment)
                   |
            ---------------------------
            |                         |
          Identity                    |
    (no target assignment)            |
            |                         |
            --------------------------
                   |
                Return
        */
    auto s = migraphx::shape{migraphx::shape::float_type, {8}};
    migraphx::target_assignments tass;
    migraphx::program p1;
    {
        auto* mm            = p1.get_main_module();
        auto x_param        = mm->add_parameter("x", s);
        auto y_param        = mm->add_parameter("y", s);
        auto add_ins        = mm->add_instruction(migraphx::make_op("add"), x_param, y_param);
        auto identity_ins_0 = mm->add_instruction(migraphx::make_op("identity"), add_ins);
        auto identity_ins_1 = mm->add_instruction(migraphx::make_op("identity"), identity_ins_0);
        mm->add_return({identity_ins_0, identity_ins_1});
        tass.insert(tass.begin(), std::make_pair(add_ins, 0));
    }
    migraphx::generate_root_modules(p1, tass);
    migraphx::program p2;
    {
        migraphx::module_ref mm             = p2.get_main_module();
        auto y                              = mm->add_parameter("y", s);
        auto x                              = mm->add_parameter("x", s);
        migraphx::module_ref target_mod_0_0 = p2.create_module("target_mod_0_0");
        auto target_mod_0_0_param_1         = target_mod_0_0->add_parameter("param:1", s);
        auto target_mod_0_0_param_0         = target_mod_0_0->add_parameter("param:0", s);
        auto x_target_mod_0_0_2             = target_mod_0_0->add_instruction(
            migraphx::make_op("add"), target_mod_0_0_param_1, target_mod_0_0_param_0);
        target_mod_0_0->add_return({x_target_mod_0_0_2});

        auto x_2 = mm->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", 0}}), {y, x}, {target_mod_0_0});
        auto x_3 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_2);
        auto x_4 = mm->add_instruction(migraphx::make_op("identity"), x_3);
        auto x_5 = mm->add_instruction(migraphx::make_op("identity"), x_4);
        mm->add_return({x_4, x_5});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(nested_if_then_else_program)
{
    /*
                                If ----------------> Return
                                |
                    -----------------------------------------
                    |                                        |
                (then_mod)                                (else_mod)
                    |                                        |
                   Add (tid = 3)                          Mul (tid = 2)
                    |                                        |
                   If                                       If
                    |                                        |
             ----------------------                --------------------
             |                     |               |                   |
        (then_mod)             (else_mod)        (then_mod)         (else_mod)
             |                     |               |                   |
        Add (tid = 1)        Add (tid = 0)      Add (tid = 0)      Add (tid = 1)
    */

    migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
    migraphx::shape cond_s{migraphx::shape::bool_type};
    migraphx::target_assignments tass;
    std::vector<float> data(ds.elements(), -1);
    migraphx::program p1;
    {
        std::unordered_map<std::size_t, std::size_t> counter_map = {{0, 0}, {1, 0}};
        auto* mm                                                 = p1.get_main_module();
        auto cond_0             = mm->add_parameter("cond_0", cond_s);
        auto cond_1             = mm->add_parameter("cond_1", cond_s);
        auto x                  = mm->add_parameter("x", ds);
        auto y                  = mm->add_parameter("y", ds);
        auto z                  = mm->add_parameter("z", ds);
        auto create_test_module = [&](migraphx::program& prog,
                                      std::size_t tid,
                                      const std::string& param_prefix) {
            std::string mod_name =
                "target_" + std::to_string(tid) + "_" + std::to_string(counter_map[tid]++);
            auto* test_mod        = prog.create_module(mod_name);
            auto l1               = test_mod->add_literal(migraphx::literal(ds, data));
            auto test_mod_param_0 = test_mod->add_parameter(param_prefix + "_param_0", ds);
            auto ins1 = test_mod->add_instruction(migraphx::make_op("add"), test_mod_param_0, l1);
            test_mod->add_return({ins1});
            tass.insert(tass.begin(), std::make_pair(ins1, tid));
            return test_mod;
        };

        auto* then_mod        = p1.create_module("then_mod");
        auto then_mod_cond    = then_mod->add_parameter("then_mod_cond", cond_s);
        auto then_mod_param_0 = then_mod->add_parameter("then_mod_param_0", ds);
        auto then_mod_param_1 = then_mod->add_parameter("then_mod_param_1", ds);
        auto then_mod_add_ins =
            then_mod->add_instruction(migraphx::make_op("add"), then_mod_param_0, then_mod_param_1);
        tass.insert(tass.begin(), std::make_pair(then_mod_add_ins, 3));
        auto then_mod_if = then_mod->add_instruction(
            migraphx::make_op("if"),
            {then_mod_cond, then_mod_param_0, then_mod_add_ins},
            {create_test_module(p1, 1, "1_"), create_test_module(p1, 0, "2_")});
        auto then_mod_if_0 = then_mod->add_instruction(
            migraphx::make_op("get_tuple_elem", {{"index", 0}}), then_mod_if);
        then_mod->add_return({then_mod_if_0});

        auto* else_mod        = p1.create_module("else_mod");
        auto else_mod_cond    = else_mod->add_parameter("else_mod_cond", cond_s);
        auto else_mod_param_0 = else_mod->add_parameter("else_mod_param_0", ds);
        auto else_mod_param_1 = else_mod->add_parameter("else_mod_param_1", ds);
        auto else_mod_add_ins =
            else_mod->add_instruction(migraphx::make_op("mul"), else_mod_param_0, else_mod_param_1);
        tass.insert(tass.begin(), std::make_pair(else_mod_add_ins, 2));
        auto else_mod_if = else_mod->add_instruction(
            migraphx::make_op("if"),
            {else_mod_cond, else_mod_add_ins, else_mod_param_0},
            {create_test_module(p1, 0, "1_"), create_test_module(p1, 1, "2_")});
        auto else_mod_if_0 = else_mod->add_instruction(
            migraphx::make_op("get_tuple_elem", {{"index", 0}}), else_mod_if);
        else_mod->add_return({else_mod_if_0});

        // Create nested and multi-target main module using "If"
        auto main_if_ins = mm->add_instruction(
            migraphx::make_op("if"), {cond_0, cond_1, x, y, cond_1, x, z}, {then_mod, else_mod});
        auto r =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), main_if_ins);
        mm->add_return({r});
    }
    migraphx::generate_root_modules(p1, tass);
    migraphx::program p2;
    {
        std::unordered_map<std::size_t, std::size_t> counter_map = {{0, 0}, {1, 0}};
        migraphx::module_ref mm                                  = p2.get_main_module();
        auto z                                                   = mm->add_parameter("z", ds);
        auto y                                                   = mm->add_parameter("y", ds);
        auto x                                                   = mm->add_parameter("x", ds);
        auto cond_1             = mm->add_parameter("cond_1", cond_s);
        auto cond_0             = mm->add_parameter("cond_0", cond_s);
        auto create_test_module = [&](migraphx::program& prog, std::size_t tid) {
            std::string mod_name =
                "target_mod_" + std::to_string(tid) + "_" + std::to_string(counter_map[tid]++);
            auto* test_mod        = prog.create_module(mod_name);
            auto test_mod_param_0 = test_mod->add_parameter("param:0", ds);
            auto test_mod_param_1 = test_mod->add_parameter("param:1", ds);
            auto ins1             = test_mod->add_instruction(
                migraphx::make_op("add"), test_mod_param_1, test_mod_param_0);
            test_mod->add_return({ins1});
            tass.insert(tass.begin(), std::make_pair(ins1, tid));
            return test_mod;
        };

        migraphx::module_ref target_1_0 = p2.create_module("target_1_0");
        auto target_1_0_1_param_0       = target_1_0->add_literal(ds, data);
        auto target_1_0_1_param_1       = target_1_0->add_parameter("1__param_0", ds);
        auto x_target_1_0_2 =
            target_1_0->add_instruction(migraphx::make_op("run_on_target", {{"target_id", 1}}),
                                        {target_1_0_1_param_0, target_1_0_1_param_1},
                                        {create_test_module(p2, 1)});
        auto x_target_1_0_3 = target_1_0->add_instruction(
            migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_target_1_0_2);
        target_1_0->add_return({x_target_1_0_3});

        migraphx::module_ref target_0_0 = p2.create_module("target_0_0");
        auto target_0_0_2_param_0       = target_0_0->add_literal(ds, data);
        auto target_0_0_2_param_1       = target_0_0->add_parameter("2__param_0", ds);
        auto x_target_0_0_2 =
            target_0_0->add_instruction(migraphx::make_op("run_on_target", {{"target_id", 0}}),
                                        {target_0_0_2_param_0, target_0_0_2_param_1},
                                        {create_test_module(p2, 0)});
        auto x_target_0_0_3 = target_0_0->add_instruction(
            migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_target_0_0_2);
        target_0_0->add_return({x_target_0_0_3});

        migraphx::module_ref target_3_0 = p2.create_module("target_mod_3_0");
        auto target_mod_3_0_param_1     = target_3_0->add_parameter("param:1", ds);
        auto target_mod_3_0_param_0     = target_3_0->add_parameter("param:0", ds);
        auto target_3_add_ins           = target_3_0->add_instruction(
            migraphx::make_op("add"), target_mod_3_0_param_1, target_mod_3_0_param_0);
        target_3_0->add_return({target_3_add_ins});

        migraphx::module_ref then_mod  = p2.create_module("then_mod");
        auto then_mod_then_mod_param_1 = then_mod->add_parameter("then_mod_param_1", ds);
        auto then_mod_then_mod_param_0 = then_mod->add_parameter("then_mod_param_0", ds);
        auto then_mod_then_mod_cond    = then_mod->add_parameter("then_mod_cond", cond_s);
        auto x_then_mod_3 =
            then_mod->add_instruction(migraphx::make_op("run_on_target", {{"target_id", 3}}),
                                      {then_mod_then_mod_param_1, then_mod_then_mod_param_0},
                                      {target_3_0});
        auto x_then_mod_4 = then_mod->add_instruction(
            migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_then_mod_3);
        auto x_then_mod_5 = then_mod->add_instruction(
            migraphx::make_op("if"),
            {then_mod_then_mod_cond, then_mod_then_mod_param_0, x_then_mod_4},
            {target_1_0, target_0_0});
        auto x_then_mod_6 = then_mod->add_instruction(
            migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_then_mod_5);
        then_mod->add_return({x_then_mod_6});

        migraphx::module_ref target_0_1 = p2.create_module("target_0_1");
        auto target_0_1_1_param_0       = target_0_1->add_literal(ds, data);
        auto target_0_1_1_param_1       = target_0_1->add_parameter("1__param_0", ds);
        auto x_target_0_1_2 =
            target_0_1->add_instruction(migraphx::make_op("run_on_target", {{"target_id", 0}}),
                                        {target_0_1_1_param_0, target_0_1_1_param_1},
                                        {create_test_module(p2, 0)});
        auto x_target_0_1_3 = target_0_1->add_instruction(
            migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_target_0_1_2);
        target_0_1->add_return({x_target_0_1_3});

        migraphx::module_ref target_1_1 = p2.create_module("target_1_1");
        auto target_1_1_2_param_0       = target_1_1->add_literal(ds, data);
        auto target_1_1_2_param_1       = target_1_1->add_parameter("2__param_0", ds);
        auto x_target_1_1_2 =
            target_1_1->add_instruction(migraphx::make_op("run_on_target", {{"target_id", 1}}),
                                        {target_1_1_2_param_0, target_1_1_2_param_1},
                                        {create_test_module(p2, 1)});
        auto x_target_1_1_3 = target_1_1->add_instruction(
            migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_target_1_1_2);
        target_1_1->add_return({x_target_1_1_3});

        migraphx::module_ref target_2_0 = p2.create_module("target_mod_2_0");
        auto target_mod_2_0_param_1     = target_2_0->add_parameter("param:1", ds);
        auto target_mod_2_0_param_0     = target_2_0->add_parameter("param:0", ds);
        auto target_2_mul_ins           = target_2_0->add_instruction(
            migraphx::make_op("mul"), target_mod_2_0_param_1, target_mod_2_0_param_0);
        target_2_0->add_return({target_2_mul_ins});

        migraphx::module_ref else_mod  = p2.create_module("else_mod");
        auto else_mod_else_mod_param_0 = else_mod->add_parameter("else_mod_param_0", ds);
        auto else_mod_else_mod_param_1 = else_mod->add_parameter("else_mod_param_1", ds);
        auto else_mod_else_mod_cond    = else_mod->add_parameter("else_mod_cond", cond_s);
        auto x_else_mod_3 =
            else_mod->add_instruction(migraphx::make_op("run_on_target", {{"target_id", 2}}),
                                      {else_mod_else_mod_param_1, else_mod_else_mod_param_0},
                                      {target_2_0});
        auto x_else_mod_4 = else_mod->add_instruction(
            migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_else_mod_3);
        auto x_else_mod_5 = else_mod->add_instruction(
            migraphx::make_op("if"),
            {else_mod_else_mod_cond, x_else_mod_4, else_mod_else_mod_param_0},
            {target_0_1, target_1_1});
        auto x_else_mod_6 = else_mod->add_instruction(
            migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_else_mod_5);
        else_mod->add_return({x_else_mod_6});

        auto x_5 = mm->add_instruction(
            migraphx::make_op("if"), {cond_0, cond_1, x, y, cond_1, x, z}, {then_mod, else_mod});
        auto x_6 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), x_5);
        mm->add_return({x_6});
    }
    EXPECT(p1.sort() == p2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
