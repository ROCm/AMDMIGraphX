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

#include <cassert>

struct test_reshape_transpose_reshape_broadcast_sub
    : verify_program<test_reshape_transpose_reshape_broadcast_sub>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x2 =
            mm->add_parameter("x2", migraphx::shape{migraphx::shape::float_type, {1, 1, 32, 1}});
        auto x1 =
            mm->add_parameter("x1", migraphx::shape{migraphx::shape::float_type, {1, 512, 16, 16}});
        auto reshape1 =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 32, 16, 16, 16}}}), x1);
        auto transpose = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 3, 4, 1, 2}}}), reshape1);
        auto reshape2 = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {1, 256, 32, 16}}}), transpose);
        auto broadcast = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 256, 32, 16}}}), x2);
        auto sub = mm->add_instruction(migraphx::make_op("sub"), reshape2, broadcast);
        mm->add_return({sub});
        return p;
    }
};
