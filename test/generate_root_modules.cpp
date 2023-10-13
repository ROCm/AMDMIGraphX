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
    p1.debug_print();
};

TEST_CASE(merge_case)
{
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
    p1.debug_print();
};

TEST_CASE(fork_and_merge_case)
{
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
    p1.debug_print();
};

int main(int argc, const char* argv[]) { test::run(argc, argv); }
