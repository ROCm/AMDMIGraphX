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

// The grouped dot with a bias gathered by the same indices and a pointwise
// operator on the biased sums: it subtracts an input with an element per
// sum and multiplies by a scale gathered by the same indices, so the kernel
// reads the bias and the scale by the entry of a row and the other input by
// the row. Some indices count from the end of the tables. Half takes the
// matrix instructions, float the fallback.
template <migraphx::shape::type_t T>
struct test_gather_dot_grouped_bias_column : verify_program<test_gather_dot_grouped_bias_column<T>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto x       = mm->add_parameter("x", {T, {40, 1, 32}});
        auto w       = mm->add_parameter("w", {T, {4, 32, 24}});
        auto bias    = mm->add_parameter("bias", {T, {4, 24}});
        auto scale   = mm->add_parameter("scale", {T, {4, 24}});
        auto y       = mm->add_parameter("y", {T, {40, 1, 24}});
        auto indices = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {40}},
                              {1, 3,  1, 1, 0, -1, 1, 3, 1, 3, 1, 1, -3, 3, 1, 3, 1,  0, 3, 1,
                               3, -4, 3, 1, 1, 3,  3, 1, 3, 1, 0, 3, 1,  3, 1, 1, -1, 1, 3, 1}});
        auto wg  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, indices);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), x, wg);
        auto bg  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), bias, indices);
        auto bu  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), bg);
        auto add = mm->add_instruction(migraphx::make_op("add"), dot, bu);
        auto sub = mm->add_instruction(migraphx::make_op("sub"), y, add);
        auto sg  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), scale, indices);
        auto su  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), sg);
        auto mul = mm->add_instruction(migraphx::make_op("mul"), sub, su);
        mm->add_return({mul});
        return p;
    }
    std::string section() const { return "gemm"; }
};

template struct test_gather_dot_grouped_bias_column<migraphx::shape::half_type>;
template struct test_gather_dot_grouped_bias_column<migraphx::shape::float_type>;
