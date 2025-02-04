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
#include <reduce.hpp>
#include <pointwise.hpp>
#include <migraphx/instruction.hpp>

template <migraphx::shape::type_t DType>
struct test_fuse_reduce_mlir : verify_program<test_fuse_reduce_mlir<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape s_x{DType, {1, 4, 512, 512}};
        migraphx::shape s_w{DType, {64, 4, 3, 3}};
        migraphx::shape s_b{DType, {64}};

        auto x    = mm->add_parameter("x", s_x);
        auto w    = mm->add_parameter("w", s_w);
        auto conv = mm->add_instruction(
            migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}}), x, w);
        auto xx    = add_pointwise(p, mm, "main:pointwise1", {conv}, squared());
        mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3}}}), conv);
        mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3}}}), xx);
        return p;
    }
    std::string section() const { return "conv"; }
};

template struct test_fuse_reduce_mlir<migraphx::shape::float_type>;
template struct test_fuse_reduce_mlir<migraphx::shape::half_type>;
