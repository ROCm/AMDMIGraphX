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

TEST_CASE(range_bounded_limit_verify_test)
{
    // 4 of the 8 elements are above 0.5, so the range is 0, 1, 2, 3 within a bound of 8
    migraphx::program p = read_onnx("range_bounded_limit_test.onnx");
    EXPECT(p.get_output_shapes().front().max_lens() == std::vector<std::size_t>{8});
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {8}};
    std::vector<float> x = {0.9, 0.1, 0.7, 0.6, 0.2, 0.0, 0.8, 0.3};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(x_shape, x.data());

    auto result = p.eval(pp).back();
    EXPECT(result.get_shape().lens() == std::vector<std::size_t>{4});
    std::vector<int64_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    EXPECT(result_vector == std::vector<int64_t>{0, 1, 2, 3});
}
