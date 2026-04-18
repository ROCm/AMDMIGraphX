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


TEST_CASE(array_feature_extractor_3d_test)
{
    // For this case, X's shape is [2, 3, 4], Y's shape is [2] (two indices)
    // Output's shape is [2, 3, 2]:
    migraphx::program p = read_onnx("array_feature_extractor_3d_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3, 4}};
    std::vector<float> x_data = {1.0,  2.0,  3.0,  4.0,
                            5.0,  6.0,  7.0,  8.0,
                            9.0,  10.0, 11.0, 12.0,
                            13.0, 14.0, 15.0, 16.0,
                            17.0, 18.0, 19.0, 20.0,
                            21.0, 22.0, 23.0, 24.0};

    migraphx::shape y_shape{migraphx::shape::int64_type, {2}};
    std::vector<int64_t> y_data = {1, 3};

    migraphx::parameter_map params;
    params["X"] = migraphx::argument(x_shape, x_data.data());
    params["Y"] = migraphx::argument(y_shape, y_data.data());
    
    auto result = p.eval(params).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // Gold output: Select columns 1 and 3 from last axis (columns of each row in both batches)
    // Batch 0: [2,4], [6,8], [10,12]
    // Batch 1: [14,16], [18,20], [22,24]
    std::vector<float> gold = {2.0, 4.0, 6.0, 8.0, 10.0, 12.0,
                                14.0, 16.0, 18.0, 20.0, 22.0, 24.0};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
