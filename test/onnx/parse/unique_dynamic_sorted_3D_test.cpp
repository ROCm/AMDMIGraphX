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

TEST_CASE(unique_dynamic_sorted_3D_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape s{migraphx::shape::int64_type, {4, 4, 4}};
    auto x = mm->add_parameter("X", s);

    auto out   = mm->add_instruction(migraphx::make_op("unique", {{"sorted", 1}}), x);
    auto y     = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), out);
    auto y_ind = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), out);
    auto x_ind = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), out);
    auto count = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 3}}), out);

    mm->add_return({y, y_ind, x_ind, count});
    auto prog = read_onnx("unique_dynamic_sorted_3D_test.onnx");

    EXPECT(p == prog);
}
