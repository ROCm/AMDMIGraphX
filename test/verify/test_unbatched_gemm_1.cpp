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
#include <migraphx/apply_alpha_beta.hpp>

template <migraphx::shape::type_t DType>
struct test_unbatched_gemm_1 : verify_program<test_unbatched_gemm_1<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{DType, {2, 32, 64}};
        migraphx::shape m2_shape{DType, {64, 64}};
        migraphx::shape m3_shape{DType, {2, 32, 192}};
        auto l1 = mm->add_parameter("1", m1_shape);
        auto l2 = mm->add_literal(migraphx::generate_literal(m2_shape));
        l2 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 64, 64}}}),
                                 l2);
        auto l3 = mm->add_literal(migraphx::generate_literal(m2_shape));
        l3 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 64, 64}}}),
                                 l3);
        auto l4 = mm->add_literal(migraphx::generate_literal(m2_shape));
        l4 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 64, 64}}}),
                                 l4);
        auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), l2, l3, l4);

        auto l5     = mm->add_parameter("3", m3_shape);
        float alpha = 1.0f;
        float beta  = 1.0f;
        migraphx::add_apply_alpha_beta(
            *mm, {l1, concat, l5}, migraphx::make_op("dot"), alpha, beta);
        return p;
    }
    std::string section() const { return "gemm"; }
};

template struct test_unbatched_gemm_1<migraphx::shape::float_type>;
template struct test_unbatched_gemm_1<migraphx::shape::half_type>;
template struct test_unbatched_gemm_1<migraphx::shape::bf16_type>;
template struct test_unbatched_gemm_1<migraphx::shape::fp8e4m3fnuz_type>;
template struct test_unbatched_gemm_1<migraphx::shape::fp8e5m2fnuz_type>;
template struct test_unbatched_gemm_1<migraphx::shape::fp8e4m3fn_type>;
template struct test_unbatched_gemm_1<migraphx::shape::fp8e5m2_type>;
