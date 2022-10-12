/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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

struct test_ck_gemm : verify_program<test_ck_gemm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        unsigned long m = 256; 
        unsigned long k = m;//4096; 
        unsigned long n = k;//4096;
        migraphx::shape m1_shape{migraphx::shape::half_type, {m, k}};
        migraphx::shape m2_shape{migraphx::shape::half_type, {k, n}};
        auto l1 = mm->add_parameter("1", m1_shape);
        auto l2 = mm->add_parameter("2", m2_shape);
        // migraphx::shape m1_shape{migraphx::shape::half_type, {1}};
        // migraphx::shape m2_shape{migraphx::shape::half_type, {1}};
        // auto l1 = mm->add_literal(migraphx::literal{m1_shape, {1}});
        // auto l2 = mm->add_literal(migraphx::literal{m2_shape, {1}});
        // l1 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {m, k}}}), l1);
        // l2 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {k, n}}}), l2);

        mm->add_instruction(migraphx::make_op("ck_gemm"), l1, l2);

        return p;
    }
};

// struct test_ck_gemm : verify_program<test_ck_gemm>
// {
//     migraphx::program create_program() const
//     {
//         migraphx::program p;
//         auto* mm = p.get_main_module();
//         unsigned long m = 3; unsigned long k = 3; unsigned long n = 3;
//         migraphx::shape m1_shape{migraphx::shape::half_type, {m, k}};
//         migraphx::shape m2_shape{migraphx::shape::half_type, {k, n}};
//         std::vector<float> v1(m * k, 1);
//         //std::iota(v1.begin(), v1.end(), 1);
//         std::vector<float> v2(k * n, 1);
//          std::iota(v2.begin(), v2.end(), 1);
//         auto l1 = mm->add_literal(migraphx::literal{m1_shape, v1});
//         auto l2 = mm->add_literal(migraphx::literal{m2_shape, v2});
//         // auto l1 = mm->add_parameter("1", m1_shape);
//         // auto l2 = mm->add_parameter("2", m2_shape);
//         // l1 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l1);
//         // l2 = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), l2);

//         mm->add_instruction(migraphx::make_op("ck_gemm"), l1, l2);

//         return p;
//     }
// };
