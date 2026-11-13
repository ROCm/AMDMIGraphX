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

TEST_CASE(matmulinteger_one_zero_zp_vec_test2)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l0 = mm->add_parameter("1", migraphx::shape{migraphx::shape::uint8_type, {4, 3}});
    auto l1 = mm->add_parameter("2", migraphx::shape{migraphx::shape::int8_type, {3, 2}});

    // Shift uint8 input
    auto int8_shift2 =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {128}});

    // Shift uint8 input
    auto wide_input = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), l0);

    auto mbr2 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 3}}}),
                                    int8_shift2);

    auto shifted_input = mm->add_instruction(migraphx::make_op("sub"), wide_input, mbr2);

    l0 = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}), shifted_input);

    mm->add_instruction(migraphx::make_op("quant_dot"), l0, l1);

    auto prog = optimize_onnx("matmulinteger_int8_uint8_one_zp_zero_vec_test2.onnx");

    migraphx::run_passes(prog, {migraphx::dead_code_elimination{}});
    EXPECT(p.sort() == prog.sort());
}
