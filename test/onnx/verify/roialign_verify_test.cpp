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

TEST_CASE(roialign_verify_test)
{
    migraphx::program p = read_onnx("roialign_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {3, 2, 4, 5}};
    std::vector<float> data(3*5*4*2);
    std::iota(data.begin(), data.end(), 0);

    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s, data.data());
    pp["y"]     = migraphx::argument(s, data.data());

    migraphx::shape srois{migraphx::shape::float_type, {2, 4}};
    std::vector<float> rois_data = {0.1, 0.15, 0.6, 0.35,
                                    2.1, 1.73, 3.8, 2.13};
    migraphx::shape sbi{migraphx::shape::int64_type, {2}};  // batch_index
    std::vector<float> bi_data = {0, 1};

    pp["rois"]    = migraphx::argument(srois, rois_data.data());
    pp["batch_ind"]    = migraphx::argument(sbi, bi_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

printf(" result:  ");
for(auto aa : result_vector) printf(" %f ", aa);
printf("\n");

    std::vector<float> gold = {   0.000000,  0.022222,  0.200000,  0.400000,  0.600000,  0.500000,  0.522222,  0.700000,  0.900000,  1.100000,  1.500000,  1.522223,  1.700000,
      1.900000, 2.100000, 2.500000, 2.522222, 2.700000, 2.900000, 3.100000, 3.500000, 3.522222, 3.700000, 3.900000, 4.100000, 20.000000, 20.022223, 20.200001, 20.400000, 20.600000, 20.500000, 20.522223, 
      20.700001, 20.900000, 21.100000, 21.500000, 21.522223, 21.700001, 21.900000, 22.100000, 22.500000, 22.522223, 22.700001, 22.900000, 23.100000, 23.500000, 23.522223, 23.700001, 
      23.900000, 24.100000, 5.888889, 0.000000, 0.000000, 0.000000, 0.000000, 6.000000, 0.000000, 0.000000, 0.000000, 0.000000, 6.000000, 0.000000, 0.000000, 0.000000, 0.000000,
    6.000000, 0.000000, 0.000000, 0.000000, 0.000000, 6.000000, 0.000000, 0.000000, 0.000000, 0.000000, 12.555555, 0.000000, 0.000000, 0.000000, 0.000000, 12.666667, 0.000000,
        0.000000, 0.000000, 0.000000, 12.666667, 0.000000, 0.000000, 0.000000, 0.000000, 12.666667, 0.000000, 0.000000, 0.000000, 0.000000, 12.666667, 0.000000, 0.000000,
        0.000000,  0.000000 };
    float alpha = 0.5;
    std::transform(data.begin(), data.end(), gold.begin(), [&](auto x) {
        return std::max(0.0f, x) + std::min(0.0f, alpha * std::expm1(x / alpha));
    });
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
