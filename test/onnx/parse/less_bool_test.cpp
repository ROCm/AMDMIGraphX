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

TEST_CASE(less_bool_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape sf{migraphx::shape::float_type, {2, 3}};
    migraphx::shape sb{migraphx::shape::bool_type, {2, 3}};

    auto input1 = mm->add_parameter("x1", sf);
    auto input2 = mm->add_parameter("x2", sb);
    auto cin1   = mm->add_instruction(
        migraphx::make_op("convert",
                            {{"target_type", migraphx::to_value(migraphx::shape::bool_type)}}),
        input1);
    auto ret = mm->add_instruction(migraphx::make_op("less"), cin1, input2);
    mm->add_return({ret});

    auto prog = migraphx::parse_onnx("less_bool_test.onnx");
    EXPECT(p == prog);
}
