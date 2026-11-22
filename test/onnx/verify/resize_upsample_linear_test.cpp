/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

TEST_CASE(resize_upsample_linear_test)
{
    migraphx::program p = read_onnx("resize_upsample_linear_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    std::vector<float> dx = {1.0f, 2.0f, 3.0f, 4.0f};

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        1, 1.25, 1.75, 2, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3, 3.25, 3.75, 4};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(resize_linear_after_transpose_test)
{
    auto p = read_onnx("resize_linear_after_transpose_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape input_shape{migraphx::shape::float_type, {1, 2, 2, 2}};
    std::vector<float> input = {1, 11, 2, 12, 3, 13, 4, 14};
    migraphx::parameter_map params;
    params["X"] = migraphx::argument(input_shape, input.data());

    auto result = p.eval(params).back();
    std::vector<float> actual;
    result.visit([&](auto output) { actual.assign(output.begin(), output.end()); });

    std::vector<float> expected = {1, 1.25, 1.75, 2, 1.5, 1.75, 2.25, 2.5,
                                   2.5, 2.75, 3.25, 3.5, 3, 3.25, 3.75, 4,
                                   11, 11.25, 11.75, 12, 11.5, 11.75, 12.25, 12.5,
                                   12.5, 12.75, 13.25, 13.5, 13, 13.25, 13.75, 14};
    EXPECT(migraphx::verify::verify_rms_range(actual, expected));
}
