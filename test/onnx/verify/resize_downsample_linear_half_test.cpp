/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(resize_downsample_linear_half_test)
{
    using migraphx::half;
    migraphx::program p = read_onnx("resize_downsample_linear_half_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::half_type, {1, 1, 2, 4}};
    std::vector<half> dx = {half{1}, half{2}, half{3}, half{4}, half{5}, half{6}, half{7}, half{8}};

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    auto result = p.eval(pp).back();
    std::vector<half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Expected output was calculated without any quantization
    std::vector<half> gold = {half{2.8333333}, half{4.833333}};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
