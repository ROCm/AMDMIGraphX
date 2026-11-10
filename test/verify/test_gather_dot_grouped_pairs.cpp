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
 *
 */

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

// The first layer of an MoE expert: the grouped dot, the add of the gathered
// bias, then a gated product of the two columns of every pair, as a swiglu
// over interleaved gate and linear columns. The 24 columns leave a partial
// block of pairs. Half takes the matrix instructions, float the fallback.
template <migraphx::shape::type_t T>
struct test_gather_dot_grouped_pairs : verify_program<test_gather_dot_grouped_pairs<T>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto x       = mm->add_parameter("x", {T, {40, 1, 32}});
        auto w       = mm->add_parameter("w", {T, {4, 32, 24}});
        auto bias    = mm->add_parameter("bias", {T, {4, 24}});
        auto indices = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::int32_type, {40}},
                              {1, 3, 1, 1, 0, 3, 1, 3, 1, 3, 1, 1, 3, 3, 1, 3, 1, 0, 3, 1,
                               3, 1, 3, 1, 1, 3, 3, 1, 3, 1, 0, 3, 1, 3, 1, 1, 3, 1, 3, 1}});
        auto wg  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, indices);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), x, wg);
        auto bg  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), bias, indices);
        auto bu  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), bg);
        auto add = mm->add_instruction(migraphx::make_op("add"), dot, bu);
        auto pairs =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {40, 1, 12, 2}}}), add);
        auto gate = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {1}}}), pairs);
        auto linear = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {3}}, {"starts", {1}}, {"ends", {2}}}), pairs);
        auto sig = mm->add_instruction(migraphx::make_op("sigmoid"), gate);
        auto out = mm->add_instruction(migraphx::make_op("mul"), sig, linear);
        mm->add_return({out});
        return p;
    }
    std::string section() const { return "gemm"; }
};

template struct test_gather_dot_grouped_pairs<migraphx::shape::half_type>;
template struct test_gather_dot_grouped_pairs<migraphx::shape::float_type>;
