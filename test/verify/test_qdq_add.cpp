/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <quantize_helpers.hpp>

// Quantized residual add: simplify_qdq rewrites the add to run on the quantized inputs
struct test_qdq_add : verify_program<test_qdq_add>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape sx{migraphx::shape::float_type, {2, 8, 4, 4}};
        migraphx::shape sr{migraphx::shape::uint8_type, {2, 8, 4, 4}};
        auto x      = mm->add_parameter("x", sx);
        auto r      = mm->add_parameter("r", sr);
        auto scale1 = mm->add_literal(0.02f);
        auto zp1    = mm->add_literal(std::uint8_t{10});
        auto scale2 = mm->add_literal(0.05f);
        auto zp2    = mm->add_literal(std::uint8_t{5});
        auto scale3 = mm->add_literal(0.03f);
        auto zp3    = mm->add_literal(std::uint8_t{20});

        auto q1  = add_quantize_op(*mm, "quantizelinear", x, scale1, zp1);
        auto d1  = add_quantize_op(*mm, "dequantizelinear", q1, scale1, zp1);
        auto d2  = add_quantize_op(*mm, "dequantizelinear", r, scale2, zp2);
        auto add = mm->add_instruction(migraphx::make_op("add"), d1, d2);
        auto q3  = add_quantize_op(*mm, "quantizelinear", add, scale3, zp3);
        mm->add_return({q3});
        return p;
    }
};
