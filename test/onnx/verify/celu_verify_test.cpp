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

TEST_CASE(celu_verify_test)
{
    //  ../../build/bin/test_verify_onnx celu_verify_test
    migraphx::program p = read_onnx("roialign_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {1, 1, 2, 3}};
    std::vector<float> data = {-5.5, 2.0, 100., 7.0, 0., -1.};

    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s, data.data());
    pp["y"]     = migraphx::argument(s, data.data());  // ?

        // migraphx::shape sx{migraphx::shape::float_type, {10, 5, 4, 7}};
    migraphx::shape srois{migraphx::shape::float_type, {1, 4}};
    std::vector<float> rois_data = {0.1, 0.15, 0.6, 0.35};
    migraphx::shape sbi{migraphx::shape::int64_type, {1}};  // batch_index
    std::vector<float> bi_data = {0};

    pp["rois"]    = migraphx::argument(srois, rois_data.data());
    pp["batch_ind"]    = migraphx::argument(sbi, bi_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

printf(" result:  ");
for(auto aa : result_vector) printf(" %f ", aa);
printf("\n");

    std::vector<float> gold(6);
    float alpha = 0.5;
    std::transform(data.begin(), data.end(), gold.begin(), [&](auto x) {
        return std::max(0.0f, x) + std::min(0.0f, alpha * std::expm1(x / alpha));
    });
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
