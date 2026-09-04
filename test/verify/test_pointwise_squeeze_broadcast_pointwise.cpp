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

struct test_pointwise_squeeze_broadcast_pointwise
    : verify_program<test_pointwise_squeeze_broadcast_pointwise>
{
    migraphx::program create_program() const
    {
        migraphx::shape s1{migraphx::shape::half_type, {1, 1, 1, 8}};
        migraphx::shape s2{migraphx::shape::half_type, {1, 8, 4, 4}};
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto x    = mm->add_parameter("x", s1);
        auto y    = mm->add_parameter("y", s1);
        auto z    = mm->add_parameter("z", s2);
        auto mul  = mm->add_instruction(migraphx::make_op("mul"), x, y);
        auto add1 = mm->add_instruction(migraphx::make_op("add"), mul, x);
        auto sq   = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1, 2}}}), add1);
        auto b    = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", s2.lens()}}), sq);
        auto add2 = mm->add_instruction(migraphx::make_op("add"), z, b);
        auto relu = mm->add_instruction(migraphx::make_op("leaky_relu", {{"alpha", 0.2}}), add2);
        mm->add_return({relu});
        return p;
    }
};
