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
                                    1.1, 0.73, 2.2, 1.13};
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
    0.00000000e+00, 0.00000000e+00, 4.93826950e-03,
          8.88888836e-02, 2.00000003e-01, 3.11111122e-01,
          4.22222227e-01, 5.33333302e-01, 6.44444466e-01,
         0.00000000e+00, 0.00000000e+00, 4.93826950e-03,
          8.88888836e-02, 2.00000003e-01, 3.11111122e-01,
          4.22222227e-01, 5.33333302e-01, 6.44444466e-01,
         0.00000000e+00, 0.00000000e+00, 4.93826950e-03,
          8.88888836e-02, 2.00000003e-01, 3.11111122e-01,
          4.22222227e-01, 5.33333302e-01, 6.44444466e-01,
         1.90476179e-02, 1.90476179e-02, 2.39858869e-02,
          1.07936502e-01, 2.19047621e-01, 3.30158740e-01,
          4.41269815e-01, 5.52380979e-01, 6.63492084e-01,
         1.71428561e-01, 1.71428561e-01, 1.76366836e-01,
          2.60317445e-01, 3.71428549e-01, 4.82539713e-01,
          5.93650818e-01, 7.04761863e-01, 8.15872967e-01,
         3.42857152e-01, 3.42857152e-01, 3.47795397e-01,
          4.31746036e-01, 5.42857111e-01, 6.53968275e-01,
          7.65079260e-01, 8.76190484e-01, 9.87301588e-01,
         5.14285743e-01, 5.14285743e-01, 5.19223928e-01,
          6.03174567e-01, 7.14285672e-01, 8.25396836e-01,
          9.36507940e-01, 1.04761910e+00, 1.15873003e+00,

        1.20000000e+01, 1.20000000e+01, 1.20049391e+01,
          1.20888891e+01, 1.21999998e+01, 1.23111115e+01,
          1.24222221e+01, 1.25333328e+01, 1.26444445e+01,
         1.20000000e+01, 1.20000000e+01, 1.20049391e+01,
          1.20888891e+01, 1.21999998e+01, 1.23111115e+01,
          1.24222221e+01, 1.25333328e+01, 1.26444445e+01,
         1.20000000e+01, 1.20000000e+01, 1.20049391e+01,
          1.20888891e+01, 1.21999998e+01, 1.23111115e+01,
          1.24222221e+01, 1.25333328e+01, 1.26444445e+01,
         1.20190477e+01, 1.20190477e+01, 1.20239868e+01,
          1.21079369e+01, 1.22190475e+01, 1.23301582e+01,
          1.24412699e+01, 1.25523796e+01, 1.26634922e+01,
         1.21714277e+01, 1.21714277e+01, 1.21763659e+01,
          1.22603178e+01, 1.23714285e+01, 1.24825401e+01,
          1.25936518e+01, 1.27047615e+01, 1.28158722e+01,
         1.23428583e+01, 1.23428583e+01, 1.23477964e+01,
          1.24317465e+01, 1.25428581e+01, 1.26539688e+01,
          1.27650795e+01, 1.28761902e+01, 1.29873009e+01,
         1.25142860e+01, 1.25142860e+01, 1.25192232e+01,
          1.26031752e+01, 1.27142859e+01, 1.28253975e+01,
          1.29365072e+01, 1.30476189e+01, 1.31587305e+01,


       2.41400356e+01, 2.46190472e+01, 2.51746025e+01,
          2.57301579e+01, 2.60857143e+01, 2.60857143e+01,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         2.49971752e+01, 2.54761906e+01, 2.60317459e+01,
          2.65873032e+01, 2.69428539e+01, 2.69428539e+01,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         2.60257492e+01, 2.65047607e+01, 2.70603180e+01,
          2.76158714e+01, 2.79714279e+01, 2.79714279e+01,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         2.70543232e+01, 2.75333328e+01, 2.80888901e+01,
          2.86444473e+01, 2.90000038e+01, 2.90000038e+01,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         2.80828934e+01, 2.85619030e+01, 2.91174583e+01,
          2.96730137e+01, 3.00285721e+01, 3.00285721e+01,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         2.91114635e+01, 2.95904770e+01, 3.01460342e+01,
          3.07015896e+01, 3.10571423e+01, 3.10571423e+01,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         3.01400356e+01, 3.06190453e+01, 3.11746006e+01,
          3.17301598e+01, 3.20857124e+01, 3.20857124e+01,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00,

        3.61400337e+01, 3.66190453e+01, 3.71746063e+01,
          3.77301559e+01, 3.80857124e+01, 3.80857124e+01,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         3.69971771e+01, 3.74761848e+01, 3.80317497e+01,
          3.85872993e+01, 3.89428558e+01, 3.89428558e+01,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         3.80257492e+01, 3.85047646e+01, 3.90603180e+01,
          3.96158714e+01, 3.99714279e+01, 3.99714279e+01,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         3.90543251e+01, 3.95333328e+01, 4.00888863e+01,
          4.06444435e+01, 4.10000038e+01, 4.10000038e+01,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         4.00828934e+01, 4.05619049e+01, 4.11174622e+01,
          4.16730156e+01, 4.20285721e+01, 4.20285721e+01,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         4.11114655e+01, 4.15904732e+01, 4.21460304e+01,
          4.27015839e+01, 4.30571404e+01, 4.30571404e+01,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
         4.21400299e+01, 4.26190529e+01, 4.31746025e+01,
          4.37301636e+01, 4.40857201e+01, 4.40857201e+01,
          0.00000000e+00, 0.00000000e+00, 0.00000000e+00
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
