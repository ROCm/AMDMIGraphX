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

TEST_CASE(scatternd_nonpacked_indices_test)
{
    migraphx::program p;
    auto* mm       = p.get_main_module();
    const size_t n = 16;

    std::vector<int64_t> raw_idx_vec(2 * n, 0);
    for(size_t i = 0; i < n; ++i)
        raw_idx_vec[n + i] = static_cast<int64_t>(n - 1 - i);
    auto raw_indices = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {2, n}}, raw_idx_vec});

    auto data    = mm->add_parameter("data", {migraphx::shape::float_type, {1, n}});
    auto updates = mm->add_parameter("updates", {migraphx::shape::float_type, {n}});

    auto indices =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), raw_indices);
    auto r = mm->add_instruction(migraphx::make_op("scatternd_none"), data, indices, updates);
    mm->add_return({r});

    auto prog = read_onnx("scatternd_nonpacked_indices_test.onnx");
    EXPECT(p == prog);
}
