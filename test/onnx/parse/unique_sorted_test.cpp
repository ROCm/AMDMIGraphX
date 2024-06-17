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

#include <onnx_test.hpp>

TEST_CASE(unique_sorted_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s_x{migraphx::shape::float_type, {6}};
    std::vector<float> x_data = {2, 1, 1, 3, 4, 3};
    auto x                    = mm->add_literal(migraphx::literal(s_x, x_data));

    auto out   = mm->add_instruction(migraphx::make_op("unique", {{"sorted", 1}, {"axis", 0}}), x);
    auto y     = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), out);
    auto y_idx = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), out);
    auto x_idx = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), out);
    auto count = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 3}}), out);
    mm->add_return({y, y_idx, x_idx, count});
    auto prog = read_onnx("unique_sorted_test.onnx");

    EXPECT(p == prog);
}
