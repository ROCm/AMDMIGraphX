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
#include <migraphx/simplify_dynamicquantizelinear.hpp>

TEST_CASE(dynamicquantizelinear_2d_test)
{
    auto p = migraphx::parse_onnx("dynamicquantizelinear_2d_test.onnx");
    p.compile(migraphx::make_target("ref"));

    std::vector<float> data{1.0, 2.1, 1.3, 2.5, 3.34, 4.0, 1.5, 2.6, 3.9, 4.0, 3.0, 2.345};
    migraphx::shape s_x{migraphx::shape::float_type, {3, 4}};
    migraphx::parameter_map pp;
    pp["x"]      = migraphx::argument(s_x, data.data());
    auto results = p.eval(pp);

    std::vector<uint8_t> y_results;
    results.at(0).visit([&](auto output) { y_results.assign(output.begin(), output.end()); });
    std::vector<uint8_t> y_gold = {64, 134, 83, 159, 213, 255, 96, 166, 249, 255, 191, 149};
    EXPECT(migraphx::verify::verify_rms_range(y_results, y_gold));

    std::vector<float> y_scale;
    results.at(1).visit([&](auto output) { y_scale.assign(output.begin(), output.end()); });
    std::vector<float> y_scale_gold = {0.0156862754};
    EXPECT(migraphx::verify::verify_rms_range(y_scale, y_scale_gold));

    std::vector<uint8_t> y_zpt;
    results.at(2).visit([&](auto output) { y_zpt.assign(output.begin(), output.end()); });
    std::vector<uint8_t> y_zpt_gold = {0};
    EXPECT(migraphx::verify::verify_rms_range(y_zpt, y_zpt_gold));
}

TEST_CASE(dynamicquantizelinear_2d_dot_simplify_test)
{
    auto p   = migraphx::parse_onnx("dynamicquantizelinear_2d_dot_test.onnx");
    auto p2  = migraphx::parse_onnx("dynamicquantizelinear_2d_dot_test.onnx");
    auto* mm = p.get_main_module();
    migraphx::run_passes(*mm, {migraphx::simplify_dynamicquantizelinear{}});

    p.compile(migraphx::make_target("ref"));
    p2.compile(migraphx::make_target("ref"));

    std::vector<float> data{-128, -96, -32, -16, 0, -8, 32, 64, 127};
    migraphx::shape s_x{migraphx::shape::float_type, {3, 3}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data.data());

    auto results_int8  = p.eval(pp);
    auto results_uint8 = p2.eval(pp);

    // Quantized outputs should be 128 difference between them.
    std::vector<int8_t> y_results_int8;
    results_int8.at(0).visit(
        [&](auto output) { y_results_int8.assign(output.begin(), output.end()); });
    std::vector<uint8_t> y_results_uint8;
    results_uint8.at(0).visit(
        [&](auto output) { y_results_uint8.assign(output.begin(), output.end()); });

    std::vector<int8_t> y_gold_int8   = {-128, -96, -32, -16, 0, -8, 32, 64, 127};
    std::vector<uint8_t> y_gold_uint8 = {0, 32, 96, 112, 128, 120, 160, 192, 255};
    EXPECT(migraphx::verify::verify_rms_range(y_results_uint8, y_gold_uint8));
    EXPECT(migraphx::verify::verify_rms_range(y_results_int8, y_gold_int8));

    // Scales should be uneffected by any sort of shift in zero point
    std::vector<float> y_scale_int8;
    results_int8.at(1).visit(
        [&](auto output) { y_scale_int8.assign(output.begin(), output.end()); });

    std::vector<float> y_scale_uint8;
    results_uint8.at(1).visit(
        [&](auto output) { y_scale_uint8.assign(output.begin(), output.end()); });

    EXPECT(migraphx::verify::verify_rms_range(y_scale_int8, y_scale_uint8));

    std::vector<int8_t> y_zpt_int8;
    results_int8.at(2).visit([&](auto output) { y_zpt_int8.assign(output.begin(), output.end()); });
    std::vector<int8_t> y_zpt_gold_int8 = {0};
    EXPECT(migraphx::verify::verify_rms_range(y_zpt_int8, y_zpt_gold_int8));

    std::vector<uint8_t> y_zpt_uint8;
    results_uint8.at(2).visit(
        [&](auto output) { y_zpt_uint8.assign(output.begin(), output.end()); });
    std::vector<uint8_t> y_zpt_gold_uint8 = {128};
    EXPECT(migraphx::verify::verify_rms_range(y_zpt_uint8, y_zpt_gold_uint8));
}
