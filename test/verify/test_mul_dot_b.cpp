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

template <migraphx::shape::type_t DType>
struct test_mul_dot_b : verify_program<test_mul_dot_b<DType>>

{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape as{DType, {2, 256, 32}};
        migraphx::shape bs{DType, {2, 32, 128}};
        auto b = mm->add_parameter("input", bs);
        auto lit  = mm->add_literal(migraphx::generate_literal({DType, {1, 32, 1}}));
        auto litb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", bs.lens()}}), lit);
        auto mul = mm->add_instruction(migraphx::make_op("mul"), b, litb);
        auto a   = mm->add_literal(migraphx::generate_literal(as));
        auto dot = mm->add_instruction(migraphx::make_op("dot"), a, mul);
        mm->add_return({dot});
        return p;
    }
};

template struct test_mul_dot_b<migraphx::shape::float_type>;
template struct test_mul_dot_b<migraphx::shape::half_type>;
template struct test_mul_dot_b<migraphx::shape::fp8e4m3fnuz_type>;
