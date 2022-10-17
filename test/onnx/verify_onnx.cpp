/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>
#include <vector>
#include <migraphx/literal.hpp>
#include <migraphx/operators.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ref/target.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/half.hpp>
#include "test.hpp"

TEST_CASE(averagepool_notset_test)
{
    auto p = migraphx::parse_onnx("averagepool_notset_test.onnx");
    p.compile(migraphx::ref::target{});
    std::vector<float> data_x = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    migraphx::shape s_x{migraphx::shape::float_type, {1, 1, 5, 5}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {12};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(averagepool_nt_cip_test)
{
    auto p = migraphx::parse_onnx("averagepool_nt_cip_test.onnx");
    p.compile(migraphx::ref::target{});
    std::vector<float> data_x = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    migraphx::shape s_x{migraphx::shape::float_type, {1, 1, 5, 5}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {8.33333};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(batch_norm_flat_test)
{
    migraphx::program p = migraphx::parse_onnx("batch_norm_flat_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape x_shape{migraphx::shape::float_type, {10}};
    migraphx::shape c_shape(migraphx::shape::float_type, {1});
    std::vector<float> x_data        = {1.6524342,
                                 -0.51048076,
                                 0.32543048,
                                 2.4410043,
                                 2.0833702,
                                 0.44981122,
                                 1.0044622,
                                 -0.24006313,
                                 -0.43065986,
                                 0.07626268};
    std::vector<float> scale_data    = {-0.02927135};
    std::vector<float> bias_data     = {0.42347777};
    std::vector<float> mean_data     = {-0.00449735};
    std::vector<float> variance_data = {0.5184545};

    migraphx::parameter_map params;
    params["x"]        = migraphx::argument(x_shape, x_data.data());
    params["scale"]    = migraphx::argument(c_shape, scale_data.data());
    params["bias"]     = migraphx::argument(c_shape, bias_data.data());
    params["mean"]     = migraphx::argument(c_shape, mean_data.data());
    params["variance"] = migraphx::argument(c_shape, variance_data.data());

    auto result = p.eval(params).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.35612,
                               0.44404706,
                               0.4100655,
                               0.32406294,
                               0.33860153,
                               0.40500915,
                               0.38246143,
                               0.43305403,
                               0.4408022,
                               0.42019472};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(batch_norm_rank_2_test)
{
    migraphx::program p = migraphx::parse_onnx("batch_norm_rank_2_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 5}};
    migraphx::shape c_shape(migraphx::shape::float_type, {5});
    std::vector<float> x_data = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    std::vector<float> scale_data(5, 1.);
    std::vector<float> bias_data(5, 0.);
    std::vector<float> mean_data = {1., 2., 1., 2., 1.};
    std::vector<float> variance_data(5, 0.5);

    migraphx::parameter_map params;
    params["x"]        = migraphx::argument(x_shape, x_data.data());
    params["scale"]    = migraphx::argument(c_shape, scale_data.data());
    params["bias"]     = migraphx::argument(c_shape, bias_data.data());
    params["mean"]     = migraphx::argument(c_shape, mean_data.data());
    params["variance"] = migraphx::argument(c_shape, variance_data.data());

    auto result = p.eval(params).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.,
                               0.,
                               2.8284243,
                               2.8284243,
                               5.65684859,
                               7.07106074,
                               7.07106074,
                               9.89948504,
                               9.89948504,
                               12.72790933};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(batch_norm_1d_test)
{
    migraphx::program p = migraphx::parse_onnx("batch_norm_1d_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape x_shape{migraphx::shape::half_type, {2, 3, 4}};
    migraphx::shape c_shape(migraphx::shape::float_type, {3});
    std::vector<float> tmp = {1.652,     -0.5103, 0.3254,  2.441,   2.084,    0.4497,
                              1.005,     -0.2401, -0.4307, 0.07623, -0.02927, 0.4236,
                              -0.004498, -0.4282, -0.5527, 0.02205, -1.472,   -1.7295,
                              0.796,     0.9507,  0.2312,  0.664,   -0.06964, 1.035};
    std::vector<migraphx::half> x_data{tmp.cbegin(), tmp.cend()};
    std::vector<float> scale_data    = {-1.336926, -1.0679098, 0.10368501};
    std::vector<float> bias_data     = {0.20240043, -0.70175606, -0.8859727};
    std::vector<float> mean_data     = {0.30854642, -0.36574763, -0.9463552};
    std::vector<float> variance_data = {0.43428132, 0.97773486, 0.30332062};

    migraphx::parameter_map params;
    params["x"]        = migraphx::argument(x_shape, x_data.data());
    params["scale"]    = migraphx::argument(c_shape, scale_data.data());
    params["bias"]     = migraphx::argument(c_shape, bias_data.data());
    params["mean"]     = migraphx::argument(c_shape, mean_data.data());
    params["variance"] = migraphx::argument(c_shape, variance_data.data());

    auto result = p.eval(params).back();
    std::vector<migraphx::half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    tmp = {-2.523, 1.863,   0.1681,  -4.125, -3.348, -1.582, -2.182,  -0.8374,
           -0.789, -0.6934, -0.7134, -0.628, 0.8374, 1.697,  1.949,   0.7837,
           0.4927, 0.771,   -1.956,  -2.123, -0.664, -0.583, -0.7207, -0.5127};

    std::vector<migraphx::half> gold{tmp.cbegin(), tmp.cend()};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(batch_norm_2d_test)
{
    migraphx::program p = migraphx::parse_onnx("batch_norm_2d_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 3, 4, 4}};
    migraphx::shape c_shape(migraphx::shape::float_type, {3});
    std::vector<float> x_data = {
        1.6524342,   -0.51048076, 0.32543048,  2.4410043,   2.0833702,   0.44981122,  1.0044622,
        -0.24006313, -0.43065986, 0.07626268,  -0.02927135, 0.42347777,  -0.00449735, -0.4281568,
        -0.5527635,  0.02204161,  -1.4719028,  -1.7298799,  0.79596406,  0.9505461,   0.23115851,
        0.6639593,   -0.06963254, 1.0348768,   -1.336926,   -1.0679098,  0.10368501,  0.20240043,
        -0.70175606, -0.8859727,  0.30854642,  -0.36574763, -0.9463552,  0.9476916,   0.37686515,
        -0.05184272, -0.7151244,  -0.37341377, 0.59440356,  0.10051094,  -0.20755945, 0.9098465,
        1.1664004,   1.4075205,   -1.1522529,  -0.34607422, 0.32027543,  -0.6885485,  0.5404544,
        0.10012514,  0.8767704,   1.0032021,   -1.2755303,  0.23577735,  0.74239916,  1.0146079,
        0.60875916,  -0.29163074, 1.4872868,   0.20466477,  -0.26367408, -0.56394804, -0.56043875,
        0.7763664,   -0.9626441,  0.29653943,  -3.2231965,  0.03322164,  0.03402911,  0.77308357,
        -0.0654009,  -0.30463725, 0.22182712,  -0.22594836, -0.5807543,  -0.22390617, -0.24484141,
        -2.0761833,  1.8459716,   0.2455878,   0.99913245,  -0.9266217,  -0.1938893,  0.6417983,
        -1.0880078,  0.49565446,  2.1584804,   1.2276239,   3.3091128,   0.14217089,  0.9425477,
        0.07578196,  0.4067431,   0.71984154,  -0.20796849, 0.90003085};

    std::vector<float> scale_data = {0.658487, 0.03700604, 2.463201};

    std::vector<float> bias_data = {0.03497279, 0.17080553, 0.5636415};

    std::vector<float> mean_data = {0.1954783, 0.6203974, 0.8116831};

    std::vector<float> variance_data = {0.30558077, 0.04536599, 0.05461315};

    migraphx::parameter_map params;
    params["x"]        = migraphx::argument(x_shape, x_data.data());
    params["scale"]    = migraphx::argument(c_shape, scale_data.data());
    params["bias"]     = migraphx::argument(c_shape, bias_data.data());
    params["mean"]     = migraphx::argument(c_shape, mean_data.data());
    params["variance"] = migraphx::argument(c_shape, variance_data.data());

    auto result = p.eval(params).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        1.77046824e+00,  -8.05950999e-01, 1.89769119e-01,  2.70979643e+00,  2.28379035e+00,
        3.37928861e-01,  9.98617530e-01,  -4.83835101e-01, -7.10869908e-01, -1.07034385e-01,
        -2.32744321e-01, 3.06560963e-01,  -2.03234047e-01, -7.07888365e-01, -8.56317282e-01,
        -1.71621382e-01, -1.92677066e-01, -2.37493858e-01, 2.01305658e-01,  2.28160262e-01,
        1.03185430e-01,  1.78373277e-01,  5.09308279e-02,  2.42810518e-01,  -1.69228360e-01,
        -1.22493818e-01, 8.10402334e-02,  9.81894583e-02,  -5.88841513e-02, -9.08869803e-02,
        1.16629556e-01,  -5.11445105e-04, -1.79648399e+01, 1.99707508e+00,  -4.01903248e+00,
        -8.53731060e+00, -1.55278311e+01, -1.19264421e+01, -1.72633123e+00, -6.93161058e+00,
        -1.01784554e+01, 1.59821415e+00,  4.30211163e+00,  6.84334660e+00,  -2.01348572e+01,
        -1.16383028e+01, -4.61544800e+00, -1.52477398e+01, 4.45901126e-01,  -7.86099210e-02,
        8.46513629e-01,  9.97116446e-01,  -1.71726203e+00, 8.29761624e-02,  6.86453462e-01,
        1.01070285e+00,  5.27264357e-01,  -5.45261383e-01, 1.57374811e+00,  4.59154993e-02,
        -5.11959970e-01, -8.69639993e-01, -8.65459919e-01, 7.26914644e-01,  -1.04206637e-01,
        1.14543661e-01,  -4.96918678e-01, 6.87990561e-02,  6.89393356e-02,  1.97330773e-01,
        5.16659655e-02,  1.01048872e-02,  1.01564340e-01,  2.37750299e-02,  -3.78632471e-02,
        2.41298079e-02,  2.04928555e-02,  -2.97655046e-01, 3.83717060e-01,  1.05692141e-01,
        2.53922558e+00,  -1.77568626e+01, -1.00343809e+01, -1.22682428e+00, -1.94577579e+01,
        -2.76707697e+00, 1.47579327e+01,  4.94736385e+00,  2.68847847e+01,  -6.49254417e+00,
        1.94286156e+00,  -7.19223642e+00, -3.70413971e+00, -4.04303551e-01, -1.01827660e+01,
        1.49476433e+00};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(batch_norm_3d_test)
{
    migraphx::program p = migraphx::parse_onnx("batch_norm_3d_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape x_shape{migraphx::shape::half_type, {2, 2, 2, 2, 2}};
    migraphx::shape c_shape(migraphx::shape::half_type, {2});

    // using migraphx::half copy conversion since it doesn't have initializer_list constructor
    std::vector<float> tmp = {5., 5., 8., 7., 3., 4., 1., 7., 5., 5., 9., 4., 7., 2., 2., 2.,
                              6., 1., 4., 9., 2., 8., 0., 2., 1., 4., 8., 8., 3., 3., 0., 8.};
    std::vector<migraphx::half> x_data{tmp.cbegin(), tmp.cend()};
    tmp = {1., 1.};
    std::vector<migraphx::half> scale_data{tmp.cbegin(), tmp.cend()};
    tmp = {
        0.,
        0.,
    };
    std::vector<migraphx::half> bias_data{tmp.cbegin(), tmp.cend()};
    tmp = {-0.75, 0.29};
    std::vector<migraphx::half> mean_data{tmp.cbegin(), tmp.cend()};
    tmp = {0.31, 0.37};
    std::vector<migraphx::half> variance_data{tmp.cbegin(), tmp.cend()};

    migraphx::parameter_map params;
    params["x"]        = migraphx::argument(x_shape, x_data.data());
    params["scale"]    = migraphx::argument(c_shape, scale_data.data());
    params["bias"]     = migraphx::argument(c_shape, bias_data.data());
    params["mean"]     = migraphx::argument(c_shape, mean_data.data());
    params["variance"] = migraphx::argument(c_shape, variance_data.data());

    auto result = p.eval(params).back();
    std::vector<migraphx::half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    tmp = {10.33, 10.33, 15.71, 13.914, 6.734, 8.53,   3.143, 13.914, 7.742,   7.742, 14.32,
           6.098, 11.03, 2.81,  2.81,   2.81,  12.125, 3.143, 8.53,   17.52,   4.938, 15.71,
           1.347, 4.938, 1.167, 6.098,  12.67, 12.67,  4.453, 4.453,  -0.4768, 12.67};
    std::vector<migraphx::half> gold{tmp.cbegin(), tmp.cend()};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(celu_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("celu_verify_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {-5.5, 2.0, 100., 7.0, 0., -1.};

    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s, data.data());
    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> correct(6);
    float alpha = 0.5;
    std::transform(data.begin(), data.end(), correct.begin(), [&](auto x) {
        return std::max(0.0f, x) + std::min(0.0f, alpha * std::expm1(x / alpha));
    });
    EXPECT(migraphx::verify_range(result_vector, correct));
}

TEST_CASE(clip_args_type_mismatch)
{
    auto p = migraphx::parse_onnx("clip_test_args_type_mismatch.onnx");
    p.compile(migraphx::ref::target{});
    migraphx::shape s_0{migraphx::shape::float_type, {3, 3}};
    migraphx::parameter_map pp;
    std::vector<float> data_0 = {0.9, 1.2, 1.7, 1.9, 2.2, 2.7, 2.9, 3.2, 3.7};
    pp["0"]                   = migraphx::argument(s_0, data_0.data());
    auto result               = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.5, 2, 2, 1.9, 2.5, 3, 2.9, 3.2, 3.7};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(depthtospace_simple_test)
{
    auto p = migraphx::parse_onnx("depthtospace_simple_test.onnx");
    p.compile(migraphx::ref::target{});
    std::vector<float> data_in(48);
    std::iota(std::begin(data_in), std::end(data_in), 0);
    migraphx::shape s_x{migraphx::shape::float_type, {1, 8, 2, 3}};
    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s_x, data_in.data());
    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0,  12, 1,  13, 2,  14, 24, 36, 25, 37, 26, 38, 3,  15, 4,  16,
                               5,  17, 27, 39, 28, 40, 29, 41, 6,  18, 7,  19, 8,  20, 30, 42,
                               31, 43, 32, 44, 9,  21, 10, 22, 11, 23, 33, 45, 34, 46, 35, 47};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(spacetodepth_simple_test)
{
    auto p = migraphx::parse_onnx("spacetodepth_simple_test.onnx");
    p.compile(migraphx::ref::target{});
    std::vector<float> data_in(48);
    std::iota(std::begin(data_in), std::end(data_in), 0);
    migraphx::shape s_x{migraphx::shape::float_type, {1, 2, 4, 6}};
    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s_x, data_in.data());
    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {0,  2,  4,  12, 14, 16, 24, 26, 28, 36, 38, 40, 1,  3,  5,  13,
                               15, 17, 25, 27, 29, 37, 39, 41, 6,  8,  10, 18, 20, 22, 30, 32,
                               34, 42, 44, 46, 7,  9,  11, 19, 21, 23, 31, 33, 35, 43, 45, 47};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(spacetodepth_depthtospace_test)
{
    // space to depth
    auto p1 = migraphx::parse_onnx("spacetodepth_simple_test.onnx");
    p1.compile(migraphx::ref::target{});
    std::vector<float> data_in(48);
    std::iota(std::begin(data_in), std::end(data_in), 0);
    migraphx::shape s_x_1{migraphx::shape::float_type, {1, 2, 4, 6}};
    migraphx::parameter_map pp1;
    pp1["x"]     = migraphx::argument(s_x_1, data_in.data());
    auto result1 = p1.eval(pp1).back();
    // depth to space
    auto p2 = migraphx::parse_onnx("depthtospace_simple_test.onnx");
    p2.compile(migraphx::ref::target{});
    migraphx::parameter_map pp2;
    pp2["x"]     = result1;
    auto result2 = p2.eval(pp2).back();
    std::vector<float> result_vector2;
    result2.visit([&](auto output) { result_vector2.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify_range(result_vector2, data_in));
}

TEST_CASE(eyelike_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("eyelike_verify_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s{migraphx::shape::float_type, {3, 4}};
    std::vector<float> data{12, 0};
    migraphx::parameter_map pp;
    pp["T1"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> eyelike_mat = {0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.};
    EXPECT(migraphx::verify_range(result_vector, eyelike_mat));
}

TEST_CASE(eyelike_verify_negk_test)
{
    migraphx::program p = migraphx::parse_onnx("eyelike_verify_negk_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s{migraphx::shape::float_type, {3, 4}};
    std::vector<float> data{12, 0};
    migraphx::parameter_map pp;
    pp["T1"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> eyelike_mat = {0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.};
    EXPECT(migraphx::verify_range(result_vector, eyelike_mat));
}

TEST_CASE(gather_elements)
{
    migraphx::program p = migraphx::parse_onnx("gather_elements_axis0_test.onnx");
    p.compile(migraphx::ref::target{});
    migraphx::shape s_data{migraphx::shape::float_type, {3, 4}};
    std::vector<float> data = {
        0.25, 0.75, 0.9375, 0.4375, 0.6875, 0.5625, -0.875, 0.1875, -0.125, 0.5, -0.9375, -0.0625};

    migraphx::shape s_ind{migraphx::shape::int32_type, {2, 3}};
    std::vector<int> ind = {2, 1, 2, 0, 1, 0};

    migraphx::parameter_map pp;
    pp["data"]    = migraphx::argument(s_data, data.data());
    pp["indices"] = migraphx::argument(s_ind, ind.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {-0.125, 0.5625, -0.9375, 0.25, 0.5625, 0.9375};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(greaterorequal_test)
{
    migraphx::program p = migraphx::parse_onnx("greaterorequal_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data1 = {0.25, 0.75, 0.9375};
    std::vector<float> data2 = {0.25, 0.74, 0.9411};

    migraphx::parameter_map pp;
    pp["x1"] = migraphx::argument(s, data1.data());
    pp["x2"] = migraphx::argument(s, data2.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.0, 1.0, 0.0};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(hardsigmoid_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("hardsigmoid_verify_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s{migraphx::shape::float_type, {2, 5}};
    std::vector<float> data = {-10.0, -2.5, -1.0, -0.5, 0, 1.0, 2.0, 2.5, 2.6, 100.0};

    float alpha = 0.2;
    float beta  = 0.5;
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold(10);
    std::transform(data.begin(), data.end(), gold.begin(), [&](auto x) {
        return std::max(0.0f, std::min(x * alpha + beta, 1.0f));
    });
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(if_else_test)
{
    migraphx::program p = migraphx::parse_onnx("if_else_test.onnx");
    p.compile(migraphx::ref::target{});
    migraphx::shape s_data{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {0.0625, 0.75, -0.0625, 0.125, -0.125, -0.5625};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_data, data.data());
    pp["y"] = migraphx::argument(s_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        -0.0364609435, 0.475317657, -0.00417715637, -0.0599277429, 0.0755792186, -0.0218581557};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(if_literal_test)
{
    auto run_prog = [](bool cond) {
        migraphx::program p = migraphx::parse_onnx("if_literal_test.onnx");
        p.compile(migraphx::ref::target{});
        migraphx::shape s_data{migraphx::shape::bool_type};
        std::vector<char> data = {static_cast<char>(cond)};

        migraphx::parameter_map pp;
        pp["cond"] = migraphx::argument(s_data, data.data());

        auto result = p.eval(pp).back();
        std::vector<float> result_vector;
        result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

        return result_vector;
    };

    // then branch
    {
        auto result_vector      = run_prog(true);
        std::vector<float> gold = {1, 2, 3, 4, 5};
        EXPECT(migraphx::verify_range(result_vector, gold));
    }

    // else branch
    {
        auto result_vector      = run_prog(false);
        std::vector<float> gold = {5, 4, 3, 2, 1};
        EXPECT(migraphx::verify_range(result_vector, gold));
    }
}

TEST_CASE(if_pl_test)
{
    auto run_prog = [](bool cond) {
        migraphx::program p = migraphx::parse_onnx("if_pl_test.onnx");
        p.compile(migraphx::ref::target{});
        migraphx::shape xs{migraphx::shape::float_type, {2, 3}};
        migraphx::shape ys{migraphx::shape::float_type, {3, 3}};
        migraphx::shape cond_s{migraphx::shape::bool_type};

        std::vector<float> x_data(xs.elements(), 1.0f);
        std::vector<float> y_data(ys.elements(), 2.0f);
        std::vector<char> cond_data{static_cast<char>(cond)};

        migraphx::parameter_map pp;
        pp["x"]    = migraphx::argument(xs, x_data.data());
        pp["y"]    = migraphx::argument(ys, y_data.data());
        pp["cond"] = migraphx::argument(cond_s, cond_data.data());

        auto result = p.eval(pp).back();
        std::vector<float> ret;
        result.visit([&](auto output) { ret.assign(output.begin(), output.end()); });

        return ret;
    };

    // then branch
    {
        auto result_vector      = run_prog(true);
        std::vector<float> gold = {2, 3, 4, 5, 6, 7};
        EXPECT(migraphx::verify_range(result_vector, gold));
    }

    // else branch
    {
        auto result_vector      = run_prog(false);
        std::vector<float> gold = {1, 2, 3, 4, 5, 6};
        EXPECT(migraphx::verify_range(result_vector, gold));
    }
}

TEST_CASE(if_tuple_test)
{
    auto run_prog = [](bool cond) {
        migraphx::program p = migraphx::parse_onnx("if_tuple_test.onnx");
        p.compile(migraphx::ref::target{});
        migraphx::shape xs{migraphx::shape::float_type, {1, 4}};
        migraphx::shape ys{migraphx::shape::float_type, {3, 4}};
        migraphx::shape cond_s{migraphx::shape::bool_type};

        std::vector<float> x_data(xs.elements(), 1.0f);
        std::vector<float> y_data(ys.elements(), 2.0f);
        std::vector<char> cond_data{static_cast<char>(cond)};

        migraphx::parameter_map pp;
        pp["x"]    = migraphx::argument(xs, x_data.data());
        pp["y"]    = migraphx::argument(ys, y_data.data());
        pp["cond"] = migraphx::argument(cond_s, cond_data.data());

        auto results = p.eval(pp);
        std::vector<std::vector<float>> rets;
        for(const auto& arg : results)
        {
            std::vector<float> vec;
            arg.visit([&](auto output) { vec.assign(output.begin(), output.end()); });
            rets.push_back(vec);
        }

        return rets;
    };

    // then branch
    {
        auto results = run_prog(true);
        std::vector<float> gold0(4, 2.0f);
        std::vector<float> gold1(12, 4.0f);
        EXPECT(migraphx::verify_range(results.at(0), gold0));
        EXPECT(migraphx::verify_range(results.at(1), gold1));
    }

    // else branch
    {
        auto results = run_prog(false);
        std::vector<float> gold0(4, 3.0f);
        std::vector<float> gold1(12, 5.0f);
        EXPECT(migraphx::verify_range(results.at(0), gold0));
        EXPECT(migraphx::verify_range(results.at(1), gold1));
    }
}

TEST_CASE(instance_norm_test)
{
    migraphx::program p = migraphx::parse_onnx("instance_norm_val_test.onnx");

    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> result_vector(9);
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {-1.54919,
                               -1.16189,
                               -0.774596,
                               -0.387298,
                               0,
                               0.387298,
                               0.774596,
                               1.16189,
                               1.54919,
                               -2.09838,
                               -1.32379,
                               -0.549192,
                               0.225404,
                               1,
                               1.7746,
                               2.54919,
                               3.32379,
                               4.09838};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(instance_norm_3d_test)
{
    migraphx::program p = migraphx::parse_onnx("instance_norm_val_3d_test.onnx");

    p.compile(migraphx::ref::target{});
    auto result = p.eval({}).back();
    std::vector<float> result_vector(16);
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {-1.52752,
                               -1.09109,
                               -0.654653,
                               -0.218218,
                               0.218218,
                               0.654653,
                               1.09109,
                               1.52752,
                               -2.05505,
                               -1.18218,
                               -0.309306,
                               0.563565,
                               1.43644,
                               2.30931,
                               3.18218,
                               4.05505};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(lessorequal_test)
{
    migraphx::program p = migraphx::parse_onnx("lessorequal_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s{migraphx::shape::float_type, {3}};
    std::vector<float> data1 = {0.25, 0.75, 0.9375};
    std::vector<float> data2 = {0.25, 0.74, 0.9411};

    migraphx::parameter_map pp;
    pp["x1"] = migraphx::argument(s, data1.data());
    pp["x2"] = migraphx::argument(s, data2.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1, 0, 1};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(lpnormalization_1norm)
{
    migraphx::program p = migraphx::parse_onnx("lpnormalization_l1_test.onnx");
    p.compile(migraphx::ref::target{});
    migraphx::shape s{migraphx::shape::float_type, {3, 4}};
    std::vector<float> data{0.f, 2.f, -2.f, 1.f, 1.f, -5.f, 3.f, -1.f, -4.f, 3.f, 0.f, 0.f};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{0.f,
                            2.f / 5.f,
                            -2.f / 5.f,
                            1.f / 5.f,
                            1.f / 10.f,
                            -5.f / 10.f,
                            3.f / 10.f,
                            -1.f / 10.f,
                            -4.f / 7.f,
                            3.f / 7.f,
                            0.f,
                            0.f};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(lpnormalization_2norm)
{
    migraphx::program p = migraphx::parse_onnx("lpnormalization_l2_test.onnx");
    p.compile(migraphx::ref::target{});
    migraphx::shape s{migraphx::shape::float_type, {3, 4}};
    std::vector<float> data{0.f, 2.f, -2.f, 1.f, 1.f, -5.f, 3.f, -1.f, -4.f, 3.f, 0.f, 0.f};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> correct{0.f,
                               2.f / 3.f,
                               -2.f / 3.f,
                               1.f / 3.f,
                               1.f / 6.f,
                               -5.f / 6.f,
                               3.f / 6.f,
                               -1.f / 6.f,
                               -4.f / 5.f,
                               3.f / 5.f,
                               0.f,
                               0.f};
    EXPECT(migraphx::verify_range(result_vector, correct));
}

TEST_CASE(mean_broadcast_test)
{
    migraphx::program p = migraphx::parse_onnx("mean_broadcast_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s0{migraphx::shape::float_type, {1, 3, 4}};
    std::vector<float> data0(12, 1);
    migraphx::shape s1{migraphx::shape::float_type, {1, 2, 3, 4}};
    std::vector<float> data1(24, 2);
    migraphx::shape s2{migraphx::shape::float_type, {4}};
    std::vector<float> data2(4, 3);
    migraphx::shape s3{migraphx::shape::float_type, {1}};
    std::vector<float> data3(1, 4);
    migraphx::shape s4{migraphx::shape::float_type, {2, 3, 1}};
    std::vector<float> data4(6, 5);

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(s0, data0.data());
    pp["1"] = migraphx::argument(s1, data1.data());
    pp["2"] = migraphx::argument(s2, data2.data());
    pp["3"] = migraphx::argument(s3, data3.data());
    pp["4"] = migraphx::argument(s4, data4.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold(24, 3);
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(mean_test)
{
    migraphx::program p = migraphx::parse_onnx("mean_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s{migraphx::shape::double_type, {2, 2, 2}};
    const int num_elms = 8;
    const int num_data = 10;
    const std::vector<double> scalars{1.0, 2.0, -2.5, 3.3, 10.7, -1.0, 100.0, 7.9, 0.01, -56.8};
    std::vector<std::vector<double>> data;
    std::transform(scalars.begin(), scalars.end(), std::back_inserter(data), [&](const auto& i) {
        return std::vector<double>(num_elms, i);
    });

    migraphx::parameter_map pp;
    for(std::size_t i = 0; i < num_data; ++i)
        pp[std::to_string(i)] = migraphx::argument(s, data[i].data());

    auto result = p.eval(pp).back();
    std::vector<double> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    const auto mean = std::accumulate(scalars.begin(), scalars.end(), 0.0) / num_data;
    std::vector<double> gold(num_elms, mean);
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(mean_integral_test)
{
    migraphx::program p = migraphx::parse_onnx("mean_integral_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s{migraphx::shape::int32_type, {2, 2, 2}};
    const int num_elms = 8;
    const int num_data = 10;
    const std::vector<int> scalars{1, 5, 14, 2, 6, 21, 101, 0, -4, -11};
    std::vector<std::vector<int>> data;
    std::transform(scalars.begin(), scalars.end(), std::back_inserter(data), [&](const auto i) {
        return std::vector<int>(num_elms, i);
    });

    migraphx::parameter_map pp;
    for(std::size_t i = 0; i < num_data; ++i)
        pp[std::to_string(i)] = migraphx::argument(s, data[i].data());

    auto result = p.eval(pp).back();
    std::vector<double> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    const auto mean = std::accumulate(scalars.begin(), scalars.end(), 0) / num_data;
    std::vector<int> gold(num_elms, mean);
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(mod_test)
{
    migraphx::program p = migraphx::parse_onnx("mod_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s{migraphx::shape::int32_type, {3, 3, 3}};

    std::vector<int32_t> a = {-4, 7, 5,  4, -7, 8, -4, 7, 5,  4, -7, 8, -4, 7,
                              5,  4, -7, 8, -4, 7, 5,  4, -7, 8, -4, 7, 5};

    std::vector<int32_t> b = {2, -3, 8, -2, 3, 5,  2, -3, 8, -2, 3, 5,  2, -3,
                              8, -2, 3, 5,  2, -3, 8, -2, 3, 5,  2, -3, 8};

    migraphx::parameter_map p_map;
    p_map["0"] = migraphx::argument(s, a.data());
    p_map["1"] = migraphx::argument(s, b.data());

    auto result = p.eval(p_map).back();
    std::vector<int32_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int32_t> gold = {0, -2, 5, 0, 2, 3,  0, -2, 5, 0, 2, 3,  0, -2,
                                 5, 0,  2, 3, 0, -2, 5, 0,  2, 3, 0, -2, 5};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(mod_test_different_types)
{
    migraphx::program p = migraphx::parse_onnx("mod_test_different_dtypes.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s_int16{migraphx::shape::int16_type, {3, 3, 3}};
    migraphx::shape s_int32{migraphx::shape::int32_type, {3, 3, 3}};

    std::vector<int16_t> a = {-4, 7, 5,  4, -7, 8, -4, 7, 5,  4, -7, 8, -4, 7,
                              5,  4, -7, 8, -4, 7, 5,  4, -7, 8, -4, 7, 5};

    std::vector<int32_t> b = {2, -3, 8, -2, 3, 5,  2, -3, 8, -2, 3, 5,  2, -3,
                              8, -2, 3, 5,  2, -3, 8, -2, 3, 5,  2, -3, 8};

    migraphx::parameter_map p_map;
    p_map["0"] = migraphx::argument(s_int16, a.data());
    p_map["1"] = migraphx::argument(s_int32, b.data());

    auto result = p.eval(p_map).back();
    std::vector<int32_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int32_t> gold = {0, -2, 5, 0, 2, 3,  0, -2, 5, 0, 2, 3,  0, -2,
                                 5, 0,  2, 3, 0, -2, 5, 0,  2, 3, 0, -2, 5};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(mod_test_fmod)
{
    migraphx::program p = migraphx::parse_onnx("mod_test_fmod.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s{migraphx::shape::float_type, {3, 3, 3}};

    std::vector<float> a = {1.2,  -2.2, 3.3,  4.1,   -5.4,  6.7,   7.8,  -8.4, 9.9,
                            10.7, 11.2, 12.3, 13.9,  -14.2, 15.8,  16.6, 17.9, 18.2,
                            19.0, 20.0, 21.0, -22.0, 23.0,  -24.0, 25.2, 26.3, 27.1};

    std::vector<float> b = {30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
                            16, 15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4};

    migraphx::parameter_map p_map;
    p_map["0"] = migraphx::argument(s, a.data());
    p_map["1"] = migraphx::argument(s, b.data());

    auto result = p.eval(p_map).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{1.2,  -2.2, 3.3,  4.1,  -5.4,  6.7,  7.8, -8.4, 9.9,
                            10.7, 11.2, 12.3, 13.9, -14.2, 15.8, 1.6, 3.9,  5.2,
                            7.0,  9.0,  1.0,  -4.0, 7.0,   -3.0, 1.2, 1.3,  3.1};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(mod_test_fmod_different_types)
{
    migraphx::program p = migraphx::parse_onnx("mod_test_fmod_different_dtypes.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s_float{migraphx::shape::float_type, {3, 3, 3}};
    migraphx::shape s_int{migraphx::shape::int32_type, {3, 3, 3}};

    std::vector<float> a = {1.2,  -2.2, 3.3,  4.1,   -5.4,  6.7,   7.8,  -8.4, 9.9,
                            10.7, 11.2, 12.3, 13.9,  -14.2, 15.8,  16.6, 17.9, 18.2,
                            19.0, 20.0, 21.0, -22.0, 23.0,  -24.0, 25.2, 26.3, 27.1};

    std::vector<int32_t> b = {30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
                              16, 15, 14, 13, 12, 11, 10, 9,  8,  7,  6,  5,  4};

    migraphx::parameter_map p_map;
    p_map["0"] = migraphx::argument(s_float, a.data());
    p_map["1"] = migraphx::argument(s_int, b.data());

    auto result = p.eval(p_map).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{1.2,  -2.2, 3.3,  4.1,  -5.4,  6.7,  7.8, -8.4, 9.9,
                            10.7, 11.2, 12.3, 13.9, -14.2, 15.8, 1.6, 3.9,  5.2,
                            7.0,  9.0,  1.0,  -4.0, 7.0,   -3.0, 1.2, 1.3,  3.1};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(nonzero_test)
{
    migraphx::program p = migraphx::parse_onnx("nonzero_dynamic_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s{migraphx::shape::bool_type, {2, 2}};
    std::vector<char> data = {1, 1, 1, 0};

    migraphx::parameter_map pp;
    pp["data"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0, 0, 1, 0, 0, 1, 0, 0};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(resize_downsample_f_test)
{
    migraphx::program p = migraphx::parse_onnx("resize_downsample_f_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 4}};
    std::vector<float> dx(sx.elements());
    std::iota(dx.begin(), dx.end(), 0.0f);

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.0f, 3.0f};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(resize_upsample_linear_ac_test)
{
    migraphx::program p = migraphx::parse_onnx("resize_upsample_linear_ac_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    std::vector<float> dx = {1.0f, 2.0f, 3.0f, 4.0f};

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1,
                               4.0f / 3,
                               5.0f / 3,
                               2,
                               5.0f / 3,
                               2,
                               7.0f / 3,
                               8.0f / 3,
                               7.0f / 3,
                               8.0f / 3,
                               3,
                               10.0f / 3,
                               3,
                               10.0f / 3,
                               11.0f / 3,
                               4};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(resize_upsample_linear_test)
{
    migraphx::program p = migraphx::parse_onnx("resize_upsample_linear_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    std::vector<float> dx = {1.0f, 2.0f, 3.0f, 4.0f};

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        1, 1.25, 1.75, 2, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3, 3.25, 3.75, 4};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(resize_upsample_pf_test)
{
    migraphx::program p = migraphx::parse_onnx("resize_upsample_pf_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    std::vector<float> dx = {1.0f, 2.0f, 3.0f, 4.0f};

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
                               3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(reversesequence_4D_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("reversesequence_4D_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape xs{migraphx::shape::float_type, {2, 2, 2, 2}};
    std::vector<float> x_data = {
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    migraphx::parameter_map param_map;
    param_map["x"] = migraphx::argument(xs, x_data.data());

    auto result = p.eval(param_map).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        8.0, 9.0, 10.0, 11.0, 4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0, 12.0, 13.0, 14.0, 15.0};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(reversesequence_batch_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("reversesequence_batch_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape xs{migraphx::shape::float_type, {4, 4}};
    std::vector<float> x_data = {
        0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    migraphx::parameter_map param_map;
    param_map["x"] = migraphx::argument(xs, x_data.data());

    auto result = p.eval(param_map).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        0.0, 1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0, 10.0, 9.0, 8.0, 11.0, 15.0, 14.0, 13.0, 12.0};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(reversesequence_time_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("reversesequence_time_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape xs{migraphx::shape::float_type, {4, 4}};
    std::vector<float> x_data = {
        0.0, 4.0, 8.0, 12.0, 1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0};
    migraphx::parameter_map param_map;
    param_map["x"] = migraphx::argument(xs, x_data.data());

    auto result = p.eval(param_map).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        3.0, 6.0, 9.0, 12.0, 2.0, 5.0, 8.0, 13.0, 1.0, 4.0, 10.0, 14.0, 0.0, 7.0, 11.0, 15.0};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(selu_test)
{
    migraphx::program p = migraphx::parse_onnx("selu_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape xs{migraphx::shape::double_type, {2, 3}};
    std::vector<double> x_data = {1.1, 2.1, 0.0, -1.3, -5.3, 12.0};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(xs, x_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.55, 1.05, 0, -0.10912, -0.149251, 6};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(size_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("size_verify_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s{migraphx::shape::float_type, {2, 5, 3}};
    std::vector<float> data(30, 1.);
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result      = p.eval(pp).back();
    auto size_result = result.at<int64_t>();
    EXPECT(size_result == int64_t{30});
}

TEST_CASE(slice_test)
{
    migraphx::program p = migraphx::parse_onnx("slice_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape sh_data{migraphx::shape::float_type, {3, 2}};
    std::vector<float> data = {0, 1, 2, 3, 4, 5};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(sh_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {2, 3};

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(slice_5arg_test)
{
    migraphx::program p = migraphx::parse_onnx("slice_5arg_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape sh_data{migraphx::shape::float_type, {5, 5}}; // start
    std::vector<float> data = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(sh_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {10, 11, 12, 13, 15, 16, 17, 18};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(slice_reverse_test)
{
    migraphx::program p = migraphx::parse_onnx("slice_5arg_reverse_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape sh_data{migraphx::shape::float_type, {5, 5}}; // start
    std::vector<float> data = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(sh_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {14, 13, 12, 11, 19, 18, 17, 16};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(slice_step_test)
{
    migraphx::program p = migraphx::parse_onnx("slice_5arg_step_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape sh_data{migraphx::shape::float_type, {5, 5}}; // start
    std::vector<float> data = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(sh_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {14, 12};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(softplus_test)
{
    migraphx::program p = migraphx::parse_onnx("softplus_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s{migraphx::shape::float_type, {5}};
    std::vector<float> data = {0, 1, 2, 3, 4};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold(5);
    std::transform(
        data.begin(), data.end(), gold.begin(), [](auto x) { return std::log1p(std::exp(x)); });

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(softsign_test)
{
    migraphx::program p = migraphx::parse_onnx("softsign_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape s{migraphx::shape::float_type, {5}};
    std::vector<float> data = {0, 1, 2, 3, 4};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold(5);
    std::transform(
        data.begin(), data.end(), gold.begin(), [](auto x) { return x / (1.0 + std::abs(x)); });

    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(upsample_test)
{
    migraphx::program p = migraphx::parse_onnx("upsample_test.onnx");

    std::vector<float> x_data = {1, 2, 3, 4};
    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, x_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
                               3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

TEST_CASE(where_test)
{
    migraphx::program p = migraphx::parse_onnx("where_test.onnx");
    p.compile(migraphx::ref::target{});

    migraphx::shape c_shape{migraphx::shape::bool_type, {2}};
    std::vector<int8_t> c_data = {1, 0};

    migraphx::shape x_shape{migraphx::shape::float_type, {2, 2, 2}};
    std::vector<float> x_data(8, 1.0f);

    migraphx::shape y_shape{migraphx::shape::float_type, {2, 1, 2, 2}};
    std::vector<float> y_data(8, 2.0f);

    migraphx::parameter_map pp;
    pp["c"] = migraphx::argument(c_shape, c_data.data());
    pp["x"] = migraphx::argument(x_shape, x_data.data());
    pp["y"] = migraphx::argument(y_shape, y_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f,
                               1.0f,
                               2.0f};
    EXPECT(migraphx::verify_range(result_vector, gold));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
