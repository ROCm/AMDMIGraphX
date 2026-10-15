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

#include <onnx_test.hpp>

TEST_CASE(nvfp4_dq_matmul_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    // fp4 initializers are packed along the last dimension
    std::vector<uint8_t> w_bits(32, 0xd2);
    auto w =
        mm->add_literal(migraphx::literal{{migraphx::shape::fp4x2_type, {32, 1}}, w_bits.data()});
    w      = mm->add_instruction(migraphx::make_op("unpack_fp4"), w);
    auto s = mm->add_literal(migraphx::literal{{migraphx::shape::fp8e4m3fn_type, {2, 2}},
                                               std::vector<float>{1.0f, 2.0f, 0.5f, 4.0f}});
    auto s2 =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {0.5f}});
    auto x = mm->add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 32}});

    s2 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2}}}), s2);
    auto scale = mm->add_instruction(migraphx::make_op("dequantizelinear"), s, s2);
    scale      = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), scale);
    scale =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 16, 2}}}), scale);
    scale   = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {32, 2}}}), scale);
    auto dq = mm->add_instruction(migraphx::make_op("dequantizelinear"), w, scale);
    mm->add_instruction(migraphx::make_op("dot"), x, dq);

    auto prog = optimize_onnx("nvfp4_dq_matmul_test.onnx");
    EXPECT(p == prog);
}
