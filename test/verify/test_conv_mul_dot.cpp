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
struct test_dot_mul_dot : verify_program<test_dot_mul_dot<DType>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape as{DType, {4, 14, 122, 122}};
        migraphx::shape bs{DType, {4, 56, 122, 122}};
        migraphx::shape cs{DType, {56, 14, 1, 1}};
        migraphx::shape ds{DType, {4, 56, 122, 56}};

        auto a   = mm->add_parameter("a", as);
        auto b   = mm->add_parameter("b", bs);
        auto c   = mm->add_parameter("c", cs);
        auto d   = mm->add_parameter("d", ds);
        auto conv = mm->add_instruction(migraphx::make_op("convolution"), a, b);
        auto mul = mm->add_instruction(migraphx::make_op("mul"), conv, c);
        auto dot = mm->add_instruction(migraphx::make_op("dot"), mul, d);
        mm->add_return({dot});
        return p;
    }
};

template struct test_dot_mul_dot<migraphx::shape::half_type>;
template struct test_dot_mul_dot<migraphx::shape::bf16_type>;
