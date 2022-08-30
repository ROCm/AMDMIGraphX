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

struct gemm_2args_vbm : verify_program<gemm_2args_vbm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape m1_shape{migraphx::shape::float_type, {5}};
        migraphx::shape m2_shape{migraphx::shape::float_type, {2, 2, 5, 4}};
        auto l1   = mm->add_parameter("1", m1_shape);
        auto ul1  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), l1);
        auto bul1 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 1, 5}}}), ul1);

        auto l2 = mm->add_parameter("2", m2_shape);

        auto res = mm->add_instruction(migraphx::make_op("dot"), bul1, l2);
        mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), res);

        return p;
    }
};
