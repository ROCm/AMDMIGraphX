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

TEST_CASE(slice_gather_end_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto counts = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {2}}, {5, 5}});
    mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto data  = mm->add_parameter("data", {migraphx::shape::float_type, {8, 4}});
    auto index = mm->add_parameter("index", migraphx::shape{migraphx::shape::int64_type});
    auto count = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), counts, index);
    mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), count);
    // Every count is 5, so the end is known and the slice is static
    auto ret = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {5}}}), data);
    mm->add_return({ret});

    EXPECT(p == read_onnx("slice_gather_end_test.onnx"));
}
