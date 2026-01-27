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

template <migraphx::shape::type_t DType>
struct test_ck_gemm_softmax_gemm_1 : verify_program<test_ck_gemm_softmax_gemm_1<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape a_s{DType, {1, 448, 384}};
        migraphx::shape b_s{DType, {1, 1500, 384}};
        migraphx::shape b1_s{DType, {1, 1500, 384}};

        auto a = mm->add_parameter("a", a_s);
        a      = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 448, 6, 64}}}), a);
        a = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1, 3}}}), a);
        a = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), a);

        auto b = mm->add_parameter("b", b_s);
        b      = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1500, 6, 64}}}), b);
        b = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 3, 0, 1}}}), b);
        b = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), b);

        auto b1 = mm->add_parameter("b1", b1_s);
        b1 = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1500, 6, 64}}}), b1);
        b1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {2, 0, 1, 3}}}),
                                 b1);
        b1 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), b1);

        auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), a, b);
        auto scale = mm->add_literal(
            migraphx::literal{migraphx::shape{DType, {1}}, {1.0f / std::sqrt(384.0f)}});
        scale = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", gemm1->get_shape().lens()}}), scale);
        auto scaled_gemm1 = mm->add_instruction(migraphx::make_op("mul"), gemm1, scale);
        auto softmax =
            mm->add_instruction(migraphx::make_op("softmax", {{"axis", -1}}), scaled_gemm1);
        mm->add_instruction(migraphx::make_op("dot"), softmax, b1);

        return p;
    }
};

template struct test_ck_gemm_softmax_gemm_1<migraphx::shape::half_type>;
