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

#include <algorithm>
#include <cmath>
#include <migraphx/half.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>
#include <onnx_test.hpp>

TEST_CASE(group_norm_fp16_overflow_test)
{
    using migraphx::half;
    auto p = read_onnx("group_norm_v21_fp16_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::half_type, {1, 4, 2}};
    migraphx::shape scale_shape{migraphx::shape::half_type, {4}};
    std::vector<half> x_data{
        half{300}, half{-300}, half{300}, half{-300}, half{-300}, half{300}, half{-300}, half{300}};
    std::vector<half> scale{half{1}, half{1}, half{1}, half{1}};
    std::vector<half> bias{half{0}, half{0}, half{0}, half{0}};
    migraphx::parameter_map params;
    params["x"]     = migraphx::argument(x_shape, x_data.data());
    params["scale"] = migraphx::argument(scale_shape, scale.data());
    params["bias"]  = migraphx::argument(scale_shape, bias.data());

    auto result = p.eval(params).back();
    std::vector<half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<half> gold{
        half{1}, half{-1}, half{1}, half{-1}, half{-1}, half{1}, half{-1}, half{1}};
    EXPECT(std::all_of(result_vector.begin(), result_vector.end(), [](half value) {
        return std::isfinite(static_cast<float>(value));
    }));
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(group_norm_contrib_fp16_overflow_test)
{
    using migraphx::half;
    auto p = read_onnx("group_norm_contrib_3d_channel_last_half_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::half_type, {1, 4, 2}};
    migraphx::shape scale_shape{migraphx::shape::half_type, {2}};
    std::vector<half> x_data{
        half{300}, half{-300}, half{-300}, half{300}, half{300}, half{-300}, half{-300}, half{300}};
    std::vector<half> gamma{half{1}, half{1}};
    std::vector<half> beta{half{0}, half{0}};
    migraphx::parameter_map params;
    params["x"]     = migraphx::argument(x_shape, x_data.data());
    params["gamma"] = migraphx::argument(scale_shape, gamma.data());
    params["beta"]  = migraphx::argument(scale_shape, beta.data());

    auto result = p.eval(params).back();
    std::vector<half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<half> gold{
        half{1}, half{-1}, half{-1}, half{1}, half{1}, half{-1}, half{-1}, half{1}};
    EXPECT(std::all_of(result_vector.begin(), result_vector.end(), [](half value) {
        return std::isfinite(static_cast<float>(value));
    }));
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(group_norm_small_eps_fp16_test)
{
    using migraphx::half;
    auto p = read_onnx("group_norm_small_eps_half_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::half_type, {1, 4, 2}};
    migraphx::shape scale_shape{migraphx::shape::half_type, {4}};
    std::vector<half> x_data(8, half{256});
    std::vector<half> scale(4, half{1});
    std::vector<half> bias{half{0.25}, half{-0.5}, half{1.5}, half{-2}};
    migraphx::parameter_map params;
    params["x"]     = migraphx::argument(x_shape, x_data.data());
    params["scale"] = migraphx::argument(scale_shape, scale.data());
    params["bias"]  = migraphx::argument(scale_shape, bias.data());

    auto result = p.eval(params).back();
    std::vector<half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<half> gold{
        half{0.25}, half{0.25}, half{-0.5}, half{-0.5}, half{1.5}, half{1.5}, half{-2}, half{-2}};
    EXPECT(std::all_of(result_vector.begin(), result_vector.end(), [](half value) {
        return std::isfinite(static_cast<float>(value));
    }));
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(group_norm_v21_fp16_affine_rounding_test)
{
    using migraphx::half;
    auto p = read_onnx("group_norm_v21_fp16_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::half_type, {1, 4, 2}};
    migraphx::shape scale_shape{migraphx::shape::half_type, {4}};
    std::vector<half> x_data{
        half{-3}, half{-1}, half{1}, half{3}, half{-3}, half{-1}, half{1}, half{3}};
    std::vector<half> scale(4, half{1024});
    std::vector<half> bias(4, half{-1374});
    migraphx::parameter_map params;
    params["x"]     = migraphx::argument(x_shape, x_data.data());
    params["scale"] = migraphx::argument(scale_shape, scale.data());
    params["bias"]  = migraphx::argument(scale_shape, bias.data());

    auto result = p.eval(params).back();
    std::vector<half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<half> gold{half{-2748},
                           half{-1832},
                           half{-916},
                           half{0},
                           half{-2748},
                           half{-1832},
                           half{-916},
                           half{0}};
    EXPECT(result_vector.at(3) == half{0});
    EXPECT(result_vector.at(7) == half{0});
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(group_norm_contrib_fp16_affine_overflow_test)
{
    using migraphx::half;
    auto p = read_onnx("group_norm_contrib_3d_channel_last_half_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::half_type, {1, 4, 2}};
    migraphx::shape scale_shape{migraphx::shape::half_type, {2}};
    std::vector<half> x_data{
        half{1}, half{-1}, half{-1}, half{1}, half{1}, half{-1}, half{-1}, half{1}};
    std::vector<half> gamma{half{65504}, half{65504}};
    std::vector<half> beta{half{65504}, half{-65504}};
    migraphx::parameter_map params;
    params["x"]     = migraphx::argument(x_shape, x_data.data());
    params["gamma"] = migraphx::argument(scale_shape, gamma.data());
    params["beta"]  = migraphx::argument(scale_shape, beta.data());

    auto result = p.eval(params).back();
    std::vector<float> values;
    result.visit([&](auto output) { values.assign(output.begin(), output.end()); });
    // The affine result exceeds half range in both directions; it must not saturate.
    EXPECT(std::isinf(values.at(0)));
    EXPECT(not std::signbit(values.at(0)));
    EXPECT(std::isinf(values.at(1)));
    EXPECT(std::signbit(values.at(1)));
    EXPECT(std::isinf(values.at(4)));
    EXPECT(not std::signbit(values.at(4)));
    EXPECT(std::isinf(values.at(5)));
    EXPECT(std::signbit(values.at(5)));
    EXPECT(std::abs(values.at(2) - 0.328125f) < 0.001f);
    EXPECT(std::abs(values.at(3) + 0.328125f) < 0.001f);
    EXPECT(std::abs(values.at(6) - 0.328125f) < 0.001f);
    EXPECT(std::abs(values.at(7) + 0.328125f) < 0.001f);
}

TEST_CASE(group_norm_small_eps_fp16_nonzero_variance_test)
{
    using migraphx::half;
    auto p = read_onnx("group_norm_small_eps_half_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::half_type, {1, 4, 2}};
    migraphx::shape scale_shape{migraphx::shape::half_type, {4}};
    // Variance is 2^-24: epsilon=1e-12 preserves +/-1 after half rounding,
    // whereas the old 1e-7 epsilon clamp changes the magnitude to about 0.61.
    std::vector<half> x_data{half{0.000244140625f},
                             half{-0.000244140625f},
                             half{0.000244140625f},
                             half{-0.000244140625f},
                             half{0.000244140625f},
                             half{-0.000244140625f},
                             half{0.000244140625f},
                             half{-0.000244140625f}};
    std::vector<half> scale(4, half{1});
    std::vector<half> bias(4, half{0});
    migraphx::parameter_map params;
    params["x"]     = migraphx::argument(x_shape, x_data.data());
    params["scale"] = migraphx::argument(scale_shape, scale.data());
    params["bias"]  = migraphx::argument(scale_shape, bias.data());

    auto result = p.eval(params).back();
    std::vector<half> values;
    result.visit([&](auto output) { values.assign(output.begin(), output.end()); });
    std::vector<half> gold{
        half{1}, half{-1}, half{1}, half{-1}, half{1}, half{-1}, half{1}, half{-1}};
    EXPECT(migraphx::verify::verify_rms_range(values, gold));
}
