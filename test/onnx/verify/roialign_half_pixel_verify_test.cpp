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

// The half_pixel mode for the ROIAlign op
TEST_CASE(roialign_half_pixel_verify_test)
{
    migraphx::program p = read_onnx("roialign_half_pixel_test.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 4, 3}};
    std::vector<float> data(2 * 2 * 4 * 3);
    std::iota(data.begin(), data.end(), 0.f);
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());
    pp["y"] = migraphx::argument(s, data.data());

    migraphx::shape srois{migraphx::shape::float_type, {2, 4}};
    std::vector<float> rois_data = {1.1, 0.73, 1.7, 1.13, 1.1, 0.73, 2.6, 1.13};
    migraphx::shape sbi{migraphx::shape::int64_type, {2}}; // batch_index
    std::vector<int64_t> bi_data = {0, 1};

    pp["rois"]      = migraphx::argument(srois, rois_data.data());
    pp["batch_ind"] = migraphx::argument(sbi, bi_data.data());
    pp["y"]         = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold values were generated with onnxruntime
    std::vector<float> gold = {5.38,      5.4799995, 5.4799995, 6.58,      6.68,  6.68,
                               17.38,     17.48,     17.48,     18.58,     18.68, 18.68,
                               29.454998, 14.74,     0.,        30.654999, 15.34, 0.,
                               41.455,    20.74,     0.,        42.655003, 21.34, 0.};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
