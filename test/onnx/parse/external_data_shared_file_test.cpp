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
#include <migraphx/algorithm.hpp>
#include <migraphx/iterator_for.hpp>

TEST_CASE(external_data_shared_file_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape s{migraphx::shape::float_type, {4}};
    auto a = mm->add_literal(migraphx::literal{s, {0.0f, 1.0f, 2.0f, 3.0f}});
    auto b = mm->add_literal(migraphx::literal{s, {4.0f, 5.0f, 6.0f, 7.0f}});
    auto y = mm->add_instruction(migraphx::make_op("add"), a, b);
    mm->add_return({y});

    auto prog = read_onnx("external_data_shared_file_test.onnx");
    EXPECT(p == prog);

    // The literals are not copies: both point into one mapping of the file, at their offsets
    auto* pm = prog.get_main_module();
    std::vector<const char*> data;
    migraphx::transform_if(
        pm->begin(),
        pm->end(),
        std::back_inserter(data),
        [](const migraphx::instruction& ins) { return ins.name() == "@literal"; },
        [](const migraphx::instruction& ins) { return ins.get_literal().data(); });
    EXPECT(data.size() == 2);
    std::sort(data.begin(), data.end(), std::less<>{});
    EXPECT(data.back() == data.front() + 8192);
}
