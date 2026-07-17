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

struct test_gemm_softmax_transpose_gemm : verify_program<test_gemm_softmax_transpose_gemm>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        // Shapes from yolo12n model.8
        migraphx::shape query_shape{migraphx::shape::half_type, {1, 4, 400, 32}};
        migraphx::shape key_shape{migraphx::shape::half_type, {1, 4, 32, 400}};
        migraphx::shape value_shape{migraphx::shape::half_type, {1, 4, 32, 400}};
        migraphx::shape softmax_shape{migraphx::shape::half_type, {1, 4, 400, 400}};
        auto softmax_elements = softmax_shape.elements();
        auto q                = mm->add_parameter("q", query_shape);
        auto k                = mm->add_parameter("k", key_shape);
        auto v                = mm->add_parameter("v", value_shape);
        std::vector<float> scales(softmax_elements, 0.176776695f);
        auto scale   = mm->add_literal(migraphx::literal{softmax_shape, scales});
        auto dot1    = mm->add_instruction(migraphx::make_op("dot"), q, k);
        auto scaled  = mm->add_instruction(migraphx::make_op("mul"), dot1, scale);
        auto softmax = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), scaled);
        auto transposed_softmax = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), softmax);
        mm->add_instruction(migraphx::make_op("dot"), v, transposed_softmax);
        return p;
    }
    std::string section() const { return "gemm"; }
};
