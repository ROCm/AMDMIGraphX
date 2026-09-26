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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

struct test_reduce_multi_out : verify_program<test_reduce_multi_out>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s{migraphx::shape::float_type, {4, 8, 16}};
        auto x        = mm->add_parameter("x", s);
        auto x2       = mm->add_instruction(migraphx::make_op("mul"), x, x);
        auto rsum1    = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsum2    = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x2);
        auto rsum1_bc = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {4, 8, 1}}}), rsum1);
        auto rsum2_bc = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {4, 8, 1}}}), rsum2);
        auto add = mm->add_instruction(migraphx::make_op("add"), rsum1_bc, rsum2_bc);
        mm->add_return({add});
        return p;
    };

    std::string section() const { return "reduce"; }
};
