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

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_test_ones)
{
    migraphx::program p = migraphx::parse_onnx("softmaxcrossentropyloss_2d_no_reduction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data = {1, 1., 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629f, 1.38629f, 1.38629f, 1.38629f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_no_reduction_test_zeros)
{
    migraphx::program p = migraphx::parse_onnx("softmaxcrossentropyloss_2d_no_reduction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629f, 1.38629f, 1.38629f, 1.38629f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_sum_reduction_test_ones)
{
    migraphx::program p =
        migraphx::parse_onnx("softmaxcrossentropyloss_2d_sum_reduction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {5.5452f};
    for(auto r : result_vector)
        std::cout << r << std::endl;

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_sum_reduction_test_zeroes)
{
    migraphx::program p =
        migraphx::parse_onnx("softmaxcrossentropyloss_2d_sum_reduction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {5.5452f};
    for(auto r : result_vector)
        std::cout << r << std::endl;

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_mean_reduction_test_ones)
{
    migraphx::program p =
        migraphx::parse_onnx("softmaxcrossentropyloss_2d_mean_reduction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softmaxcrossentropyloss_2d_mean_reduction_test_zeroes)
{
    migraphx::program p =
        migraphx::parse_onnx("softmaxcrossentropyloss_2d_mean_reduction_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape score_shape{migraphx::shape::float_type, {4, 4}};
    std::vector<float> score_data = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    migraphx::shape label_shape{migraphx::shape::int32_type, {4}};
    std::vector<int32_t> label_data = {0, 3, 1, 2};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(score_shape, score_data.data());
    pp["1"] = migraphx::argument(label_shape, label_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {1.38629f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
