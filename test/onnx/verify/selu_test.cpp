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

#include <migraphx/bf16.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(selu_test)
{
    migraphx::program p = read_onnx("selu_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape xs{migraphx::shape::double_type, {2, 3}};
    std::vector<double> x_data = {1.1, 2.1, 0.0, -1.3, -5.3, 12.0};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(xs, x_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.55, 1.05, 0, -0.10912, -0.149251, 6};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(selu_default_test)
{
    migraphx::program p = read_onnx("selu_default_test.onnx");
    p.compile(migraphx::make_target("ref"));

    // Positive inputs from 17 up used to cancel to 0 or overflow to nan
    migraphx::shape xs{migraphx::shape::float_type, {8}};
    std::vector<float> x_data = {-10.0f, -1.0f, 0.0f, 1.0f, 17.0f, 20.0f, 50.0f, 89.0f};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(xs, x_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        -1.7580196f, -1.1113307f, 0.0f, 1.050701f, 17.861917f, 21.01402f, 52.53505f, 93.51239f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(selu_bf16_test)
{
    migraphx::program p = read_onnx("selu_bf16_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape xs{migraphx::shape::bf16_type, {6}};
    std::vector<float> tmp = {-1.0f, 0.0f, 1.0f, 6.0f, 8.0f, 50.0f};
    std::vector<migraphx::bf16> x_data{tmp.cbegin(), tmp.cend()};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(xs, x_data.data());

    auto result = p.eval(pp).back();
    std::vector<migraphx::bf16> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    tmp = {-1.109375f, 0.0f, 1.046875f, 6.3125f, 8.375f, 52.5f};
    std::vector<migraphx::bf16> gold{tmp.cbegin(), tmp.cend()};

    // The default bf16 tolerance (80 eps) also accepts the old zeros at 8 and 50
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold, 1));
}
