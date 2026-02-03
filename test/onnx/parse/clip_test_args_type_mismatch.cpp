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

TEST_CASE(clip_test_args_type_mismatch)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto input = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {3, 3}});
    auto min_val =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1, 3}},
                                          {1.5f, 2.5f, 3.5f}});
    auto max_val =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {3, 1}},
                                          {2, 3, 4}});

    auto min_bc = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), min_val);
    auto max_bc = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {3, 3}}}), max_val);
    auto max_converted = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), max_bc);

    auto clip = mm->add_instruction(migraphx::make_op("clip"), input, min_bc, max_converted);
    mm->add_return({clip});

    auto prog = read_onnx("clip_test_args_type_mismatch.onnx");
    EXPECT(p.sort() == prog.sort());
}
