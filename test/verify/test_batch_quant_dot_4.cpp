/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
struct test_batch_quant_dot_4 : verify_program<test_batch_quant_dot_4<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{DType, {2, 4, 6, 3}};
        migraphx::shape m2_shape{DType, {7, 2, 6, 3}};

        auto l1  = mm->add_parameter("a", m1_shape);
        auto l2  = mm->add_parameter("b", m2_shape);
        auto tl1 = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {3, 0, 1, 2}}}), l1);
        auto tl2 = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {3, 1, 2, 0}}}), l2);
        mm->add_instruction(migraphx::make_op("quant_dot"), tl1, tl2);
        return p;
    }
    std::string section() const { return "gemm"; }
};
template struct test_batch_quant_dot_4<migraphx::shape::int8_type>;
template struct test_batch_quant_dot_4<migraphx::shape::fp8e4m3fnuz_type>;
template struct test_batch_quant_dot_4<migraphx::shape::fp8e5m2fnuz_type>;
template struct test_batch_quant_dot_4<migraphx::shape::fp8e4m3fn_type>;
template struct test_batch_quant_dot_4<migraphx::shape::fp8e5m2_type>;
