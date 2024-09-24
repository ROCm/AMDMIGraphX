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

// This passes its own test but doesn't match ort version of test
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

    // migraphx::shape srois{migraphx::shape::float_type, {2, 4}};
    // std::vector<float> rois_data = {0.1, 0.15, 0.6, 0.35,
    //                                 2.1, 1.73, 3.8, 2.13};
    // migraphx::shape sbi{migraphx::shape::int64_type, {2}};  // batch_index
    // std::vector<float> bi_data = {0, 1};

    migraphx::shape srois{migraphx::shape::float_type, {1, 4}};
    std::vector<float> rois_data = {
                                    1.1, 0.73, 2.2, 1.13};
    migraphx::shape sbi{migraphx::shape::int64_type, {1}};  // batch_index
    std::vector<float> bi_data = {0};


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
        0.000000, 0.022222, 0.200000, 0.044444, 0.066667, 0.244444, 0.400000, 0.422222, 0.600000, 0.800000,
        0.822222, 1.000000, 1.200000, 1.222222, 1.400000, 12.000000, 12.022223, 12.200000, 12.044445, 12.066667,
        12.244445, 12.400000, 12.422222, 12.600000, 12.800000, 12.822222, 13.000000, 13.200000, 13.222222, 13.400000,
        0.911111, 3.200000, 6.200000, 1.911111, 4.200000, 7.200000, 2.829630, 5.022223, 8.022223, 2.000000,
        4.000000, 7.000000, 0.000000, 0.000000, 0.000000, 12.911111, 15.200000, 18.200001, 13.911111, 16.199999,  
        19.200001, 14.829630, 17.022223, 20.022223, 14.000000, 16.000000, 19.000000, 0.000000, 0.000000, 0.000000
    };

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}


// TEST_CASE(roialign_half_pixel_verify_test)
// {
//     migraphx::program p = read_onnx("roialign_half_pixel_test.onnx");
//     p.compile(migraphx::make_target("ref"));

//     migraphx::shape s{migraphx::shape::float_type, {1, 1, 2, 3}};
//     std::vector<float> data = {-5.5, 2.0, 100., 7.0, 0., -1.};

//     migraphx::parameter_map pp;
//     pp["x"]     = migraphx::argument(s, data.data());
//     pp["y"]     = migraphx::argument(s, data.data());

//         // migraphx::shape sx{migraphx::shape::float_type, {10, 5, 4, 7}};
//     migraphx::shape srois{migraphx::shape::float_type, {1, 4}};
//     std::vector<float> rois_data = {0.1, 0.15, 0.6, 0.35};
//     migraphx::shape sbi{migraphx::shape::int64_type, {1}};  // batch_index
//     std::vector<float> bi_data = {0};

//     pp["rois"]    = migraphx::argument(srois, rois_data.data());
//     pp["batch_ind"]    = migraphx::argument(sbi, bi_data.data());

//     auto result = p.eval(pp).back();
//     std::vector<float> result_vector;
//     result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

// printf(" result:  ");
// for(auto aa : result_vector) printf(" %f ", aa);
// printf("\n");

//     std::vector<float> gold(6);
//     float alpha = 0.5;
//     std::transform(data.begin(), data.end(), gold.begin(), [&](auto x) {
//         return std::max(0.0f, x) + std::min(0.0f, alpha * std::expm1(x / alpha));
//     });
//     EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
// }
