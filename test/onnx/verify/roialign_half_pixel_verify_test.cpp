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

TEST_CASE(roialign_half_pixel_verify_test)
{
    migraphx::program p = read_onnx("roialign_half_pixel_test.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 4, 3}};
    std::vector<float> data(2*2*4*3);
    std::iota(data.begin(), data.end(), 0.f);
    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s, data.data());
    pp["y"]     = migraphx::argument(s, data.data());  // ?

    migraphx::shape srois{migraphx::shape::float_type, {2, 4}};
    std::vector<float> rois_data = {
                                    0.1, 0.15, 0.6, 0.35,
                                    1.1, 0.73, 1.9, 1.13};
    migraphx::shape sbi{migraphx::shape::int64_type, {2}};  // batch_index
    std::vector<float> bi_data = {0, 1};


    pp["rois"]    = migraphx::argument(srois, rois_data.data());
    pp["batch_ind"]    = migraphx::argument(sbi, bi_data.data());
    pp["y"]     = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

printf(" result:  \n");
for(int i = 0; i < result_vector.size(); i++)
{
 printf(" %f ", result_vector[i]);
 if(i % 9 == 8)
     printf("\n");
}
printf("\n");

    std::vector<float> gold={
 0.000000, 0.000000, 0.004938, 0.088889, 0.200000, 0.311111, 0.422222, 0.533333, 0.644444,
 0.000000, 0.000000, 0.004938, 0.088889, 0.200000, 0.311111, 0.422222, 0.533333, 0.644444,
 0.000000, 0.000000, 0.004938, 0.088889, 0.200000, 0.311111, 0.422222, 0.533333, 0.644444,
 0.019048, 0.019048, 0.023986, 0.107937, 0.219048, 0.330159, 0.441270, 0.552381, 0.663492,
 0.171429, 0.171429, 0.176367, 0.260317, 0.371429, 0.482540, 0.593651, 0.704762, 0.815873,
 0.342857, 0.342857, 0.347795, 0.431746, 0.542857, 0.653968, 0.765079, 0.876190, 0.987302,
 0.514286, 0.514286, 0.519224, 0.603175, 0.714286, 0.825397, 0.936508, 1.047619, 1.158730,
 12.000000, 12.000000, 12.004938, 12.088889, 12.200000, 12.311111, 12.422222, 12.533334, 12.644444,
 12.000000, 12.000000, 12.004938, 12.088889, 12.200000, 12.311111, 12.422222, 12.533334, 12.644444,
 12.000000, 12.000000, 12.004938, 12.088889, 12.200000, 12.311111, 12.422222, 12.533334, 12.644444,
 12.019048, 12.019048, 12.023986, 12.107937, 12.219048, 12.330158, 12.441270, 12.552382, 12.663492,
 12.171429, 12.171429, 12.176367, 12.260318, 12.371428, 12.482540, 12.593651, 12.704762, 12.815873,
 12.342857, 12.342857, 12.347795, 12.431746, 12.542857, 12.653969, 12.765079, 12.876190, 12.987302,
 12.514286, 12.514286, 12.519224, 12.603174, 12.714286, 12.825397, 12.936508, 13.047619, 13.158731,
 4.840318, 5.009453, 5.051429, 5.051429, 5.051429, 5.051429, 5.051429, 1.683810, 0.000000,
 5.183175, 5.352311, 5.394286, 5.394286, 5.394286, 5.394286, 5.394286, 1.798095, 0.000000,
 5.526032, 5.695168, 5.737143, 5.737143, 5.737143, 5.737143, 5.737143, 1.912381, 0.000000,
 5.868889, 6.038025, 6.080000, 6.080000, 6.080000, 6.080000, 6.080000, 2.026667, 0.000000,
 6.211746, 6.380882, 6.422857, 6.422857, 6.422857, 6.422857, 6.422857, 2.140952, 0.000000,
 6.554603, 6.723739, 6.765714, 6.765714, 6.765714, 6.765714, 6.765714, 2.255238, 0.000000,
 6.897460, 7.066596, 7.108572, 7.108572, 7.108572, 7.108572, 7.108572, 2.369524, 0.000000,
 16.840317, 17.009453, 17.051428, 17.051428, 17.051428, 17.051428, 17.051428, 5.683809, 0.000000,
 17.183174, 17.352310, 17.394285, 17.394285, 17.394285, 17.394285, 17.394285, 5.798095, 0.000000,
 17.526031, 17.695168, 17.737143, 17.737143, 17.737143, 17.737143, 17.737143, 5.912381, 0.000000,
 17.868889, 18.038025, 18.080000, 18.080000, 18.080000, 18.080000, 18.080000, 6.026667, 0.000000,
 18.211746, 18.380882, 18.422857, 18.422857, 18.422857, 18.422857, 18.422857, 6.140953, 0.000000,
 18.554604, 18.723740, 18.765715, 18.765715, 18.765715, 18.765715, 18.765715, 6.255238, 0.000000,
 18.897461, 19.066597, 19.108572, 19.108572, 19.108572, 19.108572, 19.108572, 6.369524, 0.000000
              };

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


TEST_CASE(roialign_half_pixel_oob_verify_test)
{
    // One ROI extends outside of bounds of input array,
    // when scaled by spatial_scale
    migraphx::program p = read_onnx("roialign_half_pixel_test.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape s{migraphx::shape::float_type, {2, 2, 4, 3}};
    std::vector<float> data(2*2*4*3);
    std::iota(data.begin(), data.end(), 0.f);
    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s, data.data());
    pp["y"]     = migraphx::argument(s, data.data());  // ?

    migraphx::shape srois{migraphx::shape::float_type, {2, 4}};
    std::vector<float> rois_data = {
                                    0.1, 0.15, 0.6, 0.35,
                                    1.1, 0.73, 2.5, 1.13};
    migraphx::shape sbi{migraphx::shape::int64_type, {2}};  // batch_index
    std::vector<float> bi_data = {0, 1};


    pp["rois"]    = migraphx::argument(srois, rois_data.data());
    pp["batch_ind"]    = migraphx::argument(sbi, bi_data.data());
    pp["y"]     = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

printf(" result:  \n");
for(int i = 0; i < result_vector.size(); i++)
{
 printf(" %f ", result_vector[i]);
 if(i % 9 == 8)
     printf("\n");
}
printf("\n");

    std::vector<float> gold={
 0.000000, 0.000000, 0.004938, 0.088889, 0.200000, 0.311111, 0.422222, 0.533333, 0.644444,
 0.000000, 0.000000, 0.004938, 0.088889, 0.200000, 0.311111, 0.422222, 0.533333, 0.644444,
 0.000000, 0.000000, 0.004938, 0.088889, 0.200000, 0.311111, 0.422222, 0.533333, 0.644444,
 0.019048, 0.019048, 0.023986, 0.107937, 0.219048, 0.330159, 0.441270, 0.552381, 0.663492,
 0.171429, 0.171429, 0.176367, 0.260317, 0.371429, 0.482540, 0.593651, 0.704762, 0.815873,
 0.342857, 0.342857, 0.347795, 0.431746, 0.542857, 0.653968, 0.765079, 0.876190, 0.987302,
 0.514286, 0.514286, 0.519224, 0.603175, 0.714286, 0.825397, 0.936508, 1.047619, 1.158730,
 12.000000, 12.000000, 12.004938, 12.088889, 12.200000, 12.311111, 12.422222, 12.533334, 12.644444,
 12.000000, 12.000000, 12.004938, 12.088889, 12.200000, 12.311111, 12.422222, 12.533334, 12.644444,
 12.000000, 12.000000, 12.004938, 12.088889, 12.200000, 12.311111, 12.422222, 12.533334, 12.644444,
 12.019048, 12.019048, 12.023986, 12.107937, 12.219048, 12.330158, 12.441270, 12.552382, 12.663492,
 12.171429, 12.171429, 12.176367, 12.260318, 12.371428, 12.482540, 12.593651, 12.704762, 12.815873,
 12.342857, 12.342857, 12.347795, 12.431746, 12.542857, 12.653969, 12.765079, 12.876190, 12.987302,
 12.514286, 12.514286, 12.519224, 12.603174, 12.714286, 12.825397, 12.936508, 13.047619, 13.158731,
 4.840318, 5.009453, 5.051429, 5.051429, 5.051429, 5.051429, 5.051429, 1.683810, 0.000000,
 5.183175, 5.352311, 5.394286, 5.394286, 5.394286, 5.394286, 5.394286, 1.798095, 0.000000,
 5.526032, 5.695168, 5.737143, 5.737143, 5.737143, 5.737143, 5.737143, 1.912381, 0.000000,
 5.868889, 6.038025, 6.080000, 6.080000, 6.080000, 6.080000, 6.080000, 2.026667, 0.000000,
 6.211746, 6.380882, 6.422857, 6.422857, 6.422857, 6.422857, 6.422857, 2.140952, 0.000000,
 6.554603, 6.723739, 6.765714, 6.765714, 6.765714, 6.765714, 6.765714, 2.255238, 0.000000,
 6.897460, 7.066596, 7.108572, 7.108572, 7.108572, 7.108572, 7.108572, 2.369524, 0.000000,
 16.840317, 17.009453, 17.051428, 17.051428, 17.051428, 17.051428, 17.051428, 5.683809, 0.000000,
 17.183174, 17.352310, 17.394285, 17.394285, 17.394285, 17.394285, 17.394285, 5.798095, 0.000000,
 17.526031, 17.695168, 17.737143, 17.737143, 17.737143, 17.737143, 17.737143, 5.912381, 0.000000,
 17.868889, 18.038025, 18.080000, 18.080000, 18.080000, 18.080000, 18.080000, 6.026667, 0.000000,
 18.211746, 18.380882, 18.422857, 18.422857, 18.422857, 18.422857, 18.422857, 6.140953, 0.000000,
 18.554604, 18.723740, 18.765715, 18.765715, 18.765715, 18.765715, 18.765715, 6.255238, 0.000000,
 18.897461, 19.066597, 19.108572, 19.108572, 19.108572, 19.108572, 19.108572, 6.369524, 0.000000
              };

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


