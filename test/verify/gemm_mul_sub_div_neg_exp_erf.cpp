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
#include <migraphx/apply_alpha_beta.hpp>

struct gemm_mul_sub_div_neg_exp_erf : verify_program<gemm_mul_sub_div_neg_exp_erf>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{migraphx::shape::float_type, {1, 2, 3}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {1, 3, 4}};
        migraphx::shape m3_shape{migraphx::shape::float_type, {1, 2, 4}};
        migraphx::shape m4_shape{migraphx::shape::float_type, {1, 2, 4}};
        migraphx::shape m5_shape{migraphx::shape::float_type, {1, 2, 4}};
        auto l1 = mm->add_parameter("1", m1_shape);
        auto l2 = mm->add_parameter("2", m2_shape);
        auto l_mul = mm->add_parameter("3", m3_shape);
        auto l_sub = mm->add_parameter("4", m4_shape);
        auto l_div = mm->add_parameter("3", m5_shape);

        auto dot = mm->add_instruction(migraphx::make_op("dot"), l1, l2);
        auto mul = mm->add_instruction(migraphx::make_op("mul"), dot, l_mul);
        auto sub = mm->add_instruction(migraphx::make_op("sub"), mul, l_sub);
        auto div = mm->add_instruction(migraphx::make_op("div"), sub, l_div);
        auto neg = mm->add_instruction(migraphx::make_op("neg"), div);
        auto exp = mm->add_instruction(migraphx::make_op("exp"), neg);
        mm->add_instruction(migraphx::make_op("erf"), exp);
        return p;
    }
};
