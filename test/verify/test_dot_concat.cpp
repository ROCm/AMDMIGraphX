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

struct test_dot_concat : verify_program<test_dot_concat>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto a1 = mm->add_parameter("a1", migraphx::shape{migraphx::shape::half_type, {4, 24, 16}});
        auto b1 = mm->add_parameter("b1", migraphx::shape{migraphx::shape::half_type, {4, 16, 24}});
        auto a2 = mm->add_parameter("a2", migraphx::shape{migraphx::shape::half_type, {4, 24, 16}});
        auto b2 = mm->add_parameter("b2", migraphx::shape{migraphx::shape::half_type, {4, 16, 24}});

        auto dot1  = mm->add_instruction(migraphx::make_op("dot"), a1, b1);
        auto relu1 = mm->add_instruction(migraphx::make_op("relu"), dot1);

        auto dot2  = mm->add_instruction(migraphx::make_op("dot"), a2, b2);
        auto sqrt1 = mm->add_instruction(migraphx::make_op("sigmoid"), dot2);

        auto concat = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), relu1, sqrt1);

        mm->add_return({concat});
        return p;
    }
    std::string section() const { return "gemm"; }
};
