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

TEST_CASE(upsample_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape ss{migraphx::shape::float_type, {4}};
    mm->add_literal(migraphx::literal(ss, {1.0f, 1.0f, 2.0f, 3.0f}));

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    auto ix = mm->add_parameter("X", sx);

    migraphx::shape si{migraphx::shape::int32_type, {1, 1, 4, 6}};
    std::vector<int> ind = {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 2, 2, 2, 3, 3, 3};

    auto li  = mm->add_literal(migraphx::literal(si, ind));
    auto rsp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4}}}), ix);
    auto r   = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), rsp, li);
    mm->add_return({r});

    auto prog = read_onnx("upsample_test.onnx");

    EXPECT(p == prog);
}
