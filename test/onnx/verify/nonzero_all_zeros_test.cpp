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

#include <migraphx/register_target.hpp>
#include <onnx_test.hpp>

// An all-zero mask drives the parser's trim to a zero-length slice, which is the one bound
// sym::var(name, {0, max}) allows that no other test reaches.
TEST_CASE(nonzero_all_zeros_test)
{
    migraphx::program p = read_onnx("nonzero_dynamic_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::bool_type, {2, 2}};
    std::vector<char> data = {0, 0, 0, 0};

    migraphx::parameter_map pp;
    pp["data"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<int64_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // np.nonzero(np.zeros((2, 2))) is ((), ()), so the ONNX specification's output is [rank, 0].
    EXPECT(result_vector.empty());
    // Only the sliced axis collapses; the trim still keeps the padded buffer's row stride.
    EXPECT(result.get_shape() == migraphx::shape{migraphx::shape::int64_type, {2, 0}, {4, 1}});
}
