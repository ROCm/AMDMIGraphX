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

template <migraphx::shape::type_t DType>
struct test_gemm_2args_bmv : verify_program<test_gemm_2args_bmv<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{DType, {2, 3, 3, 5}};
        migraphx::shape m2_shape{DType, {5}};
        auto l1   = mm->add_parameter("1", m1_shape);
        auto l2   = mm->add_parameter("2", m2_shape);
        auto ul2  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), l2);
        auto bul2 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 5, 1}}}), ul2);

        mm->add_instruction(migraphx::make_op("dot"), l1, bul2);

        return p;
    }
    std::string section() const { return "gemm"; }
};

template struct test_gemm_2args_bmv<migraphx::shape::float_type>;
template struct test_gemm_2args_bmv<migraphx::shape::half_type>;
template struct test_gemm_2args_bmv<migraphx::shape::bf16_type>;
template struct test_gemm_2args_bmv<migraphx::shape::fp8e4m3fnuz_type>;
template struct test_gemm_2args_bmv<migraphx::shape::fp8e5m2fnuz_type>;
template struct test_gemm_2args_bmv<migraphx::shape::fp8e4m3fn_type>;
template struct test_gemm_2args_bmv<migraphx::shape::fp8e5m2_type>;
