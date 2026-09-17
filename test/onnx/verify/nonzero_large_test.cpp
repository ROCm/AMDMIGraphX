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
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

// Regression: the original GPU JIT NonZero kernel used a uint8_t accumulator
// inside block_scan, which wrapped at >255 set bits and silently produced
// out-of-bounds writes. 32x32 = 1024 elements, all true, comfortably exceeds
// that historical limit. On ref this just exercises the dense path; the
// matching test_nonzero_large in test/verify/ catches the GPU kernel.
TEST_CASE(nonzero_large_test)
{
    migraphx::program p = read_onnx("nonzero_large_test.onnx");
    p.compile(migraphx::make_target("ref"));

    constexpr std::size_t rows = 32;
    constexpr std::size_t cols = 32;
    constexpr std::size_t n    = rows * cols;

    migraphx::shape s{migraphx::shape::bool_type, {rows, cols}};
    std::vector<char> data(n, 1);

    migraphx::parameter_map pp;
    pp["data"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<int64_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int64_t> gold(2 * n, 0);
    for(std::size_t i = 0; i < n; ++i)
    {
        gold[i]     = static_cast<int64_t>(i / cols);
        gold[n + i] = static_cast<int64_t>(i % cols);
    }

    EXPECT(result_vector == gold);
}
