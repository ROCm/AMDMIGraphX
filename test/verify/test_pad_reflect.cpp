/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/pad.hpp>

struct test_pad_reflect : verify_program<test_pad_reflect>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s0{migraphx::shape::float_type, {2, 2}};
        std::vector<int64_t> pads0 = {0, 2, 0, 1};
        auto l0                    = mm->add_parameter("x", s0);
        mm->add_instruction(migraphx::make_op("pad", {{"pads", pads0}, {"mode", migraphx::op::pad::reflect_pad}}), l0);
        return p;
    }
};

struct test_pad_reflect_2l2r : verify_program<test_pad_reflect_2l2r>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s0{migraphx::shape::float_type, {4, 4}};
        std::vector<int64_t> pads0 = {0, 2, 0, 2};
        auto l0                    = mm->add_parameter("x", s0);
        mm->add_instruction(migraphx::make_op("pad", {{"pads", pads0}, {"mode", migraphx::op::pad::reflect_pad}}), l0);
        return p;
    }
};

struct test_pad_reflect_3l2r : verify_program<test_pad_reflect_3l2r>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s0{migraphx::shape::float_type, {4, 4}};
        std::vector<int64_t> pads0 = {0, 3, 0, 2};
        auto l0                    = mm->add_parameter("x", s0);
        mm->add_instruction(migraphx::make_op("pad", {{"pads", pads0}, {"mode", migraphx::op::pad::reflect_pad}}), l0);
        return p;
    }
};

struct test_pad_reflect_multiaxis : verify_program<test_pad_reflect_multiaxis>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s0{migraphx::shape::float_type, {4, 4}};
        std::vector<int64_t> pads0 = {0, 2, 2, 0};
        auto l0                    = mm->add_parameter("x", s0);
        mm->add_instruction(migraphx::make_op("pad", {{"pads", pads0}, {"mode", migraphx::op::pad::reflect_pad}}), l0);
        return p;
    }
};
