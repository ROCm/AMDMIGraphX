/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/onnx.hpp>
#include <migraphx/half.hpp>
#include "test.hpp"

TEST_CASE(averagepool_notset_test)
{
    auto p = migraphx::parse_onnx("averagepool_notset_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<float> data_x = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    migraphx::shape s_x{migraphx::shape::float_type, {1, 1, 5, 5}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {12};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(averagepool_nt_cip_test)
{
    auto p = migraphx::parse_onnx("averagepool_nt_cip_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<float> data_x = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    migraphx::shape s_x{migraphx::shape::float_type, {1, 1, 5, 5}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {8.33333};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(batch_norm_flat_test)
{
    migraphx::program p = migraphx::parse_onnx("batch_norm_flat_test.onnx");
    p.compile(migraphx::make_target("ref"));

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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(batch_norm_rank_2_test)
{
    migraphx::program p = migraphx::parse_onnx("batch_norm_rank_2_test.onnx");
    p.compile(migraphx::make_target("ref"));

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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(batch_norm_1d_test)
{
    migraphx::program p = migraphx::parse_onnx("batch_norm_1d_test.onnx");
    p.compile(migraphx::make_target("ref"));

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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(batch_norm_2d_test)
{
    migraphx::program p = migraphx::parse_onnx("batch_norm_2d_test.onnx");
    p.compile(migraphx::make_target("ref"));

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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(batch_norm_3d_test)
{
    migraphx::program p = migraphx::parse_onnx("batch_norm_3d_test.onnx");
    p.compile(migraphx::make_target("ref"));

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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(celu_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("celu_verify_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {-5.5, 2.0, 100., 7.0, 0., -1.};

    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s, data.data());
    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold(6);
    float alpha = 0.5;
    std::transform(data.begin(), data.end(), gold.begin(), [&](auto x) {
        return std::max(0.0f, x) + std::min(0.0f, alpha * std::expm1(x / alpha));
    });
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(clip_args_type_mismatch)
{
    auto p = migraphx::parse_onnx("clip_test_args_type_mismatch.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape s_0{migraphx::shape::float_type, {3, 3}};
    migraphx::parameter_map pp;
    std::vector<float> data_0 = {0.9, 1.2, 1.7, 1.9, 2.2, 2.7, 2.9, 3.2, 3.7};
    pp["0"]                   = migraphx::argument(s_0, data_0.data());
    auto result               = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.5, 2, 2, 1.9, 2.5, 3, 2.9, 3.2, 3.7};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(depthtospace_simple_test)
{
    auto p = migraphx::parse_onnx("depthtospace_simple_test.onnx");
    p.compile(migraphx::make_target("ref"));
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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(spacetodepth_simple_test)
{
    auto p = migraphx::parse_onnx("spacetodepth_simple_test.onnx");
    p.compile(migraphx::make_target("ref"));
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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(spacetodepth_depthtospace_test)
{
    // space to depth
    auto p1 = migraphx::parse_onnx("spacetodepth_simple_test.onnx");
    p1.compile(migraphx::make_target("ref"));
    std::vector<float> gold_data_in(48);
    std::iota(std::begin(gold_data_in), std::end(gold_data_in), 0);
    migraphx::shape s_x_1{migraphx::shape::float_type, {1, 2, 4, 6}};
    migraphx::parameter_map pp1;
    pp1["x"]     = migraphx::argument(s_x_1, gold_data_in.data());
    auto result1 = p1.eval(pp1).back();
    // depth to space
    auto p2 = migraphx::parse_onnx("depthtospace_simple_test.onnx");
    p2.compile(migraphx::make_target("ref"));
    migraphx::parameter_map pp2;
    pp2["x"]     = result1;
    auto result2 = p2.eval(pp2).back();
    std::vector<float> result_vector2;
    result2.visit([&](auto output) { result_vector2.assign(output.begin(), output.end()); });
    EXPECT(migraphx::verify::verify_rms_range(result_vector2, gold_data_in));
}

TEST_CASE(eyelike_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("eyelike_verify_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {3, 4}};
    std::vector<float> data{12, 0};
    migraphx::parameter_map pp;
    pp["T1"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold_eyelike_mat = {0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold_eyelike_mat));
}

TEST_CASE(eyelike_verify_negk_test)
{
    migraphx::program p = migraphx::parse_onnx("eyelike_verify_negk_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {3, 4}};
    std::vector<float> data{12, 0};
    migraphx::parameter_map pp;
    pp["T1"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold_eyelike_mat = {0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold_eyelike_mat));
}

TEST_CASE(gather_elements)
{
    migraphx::program p = migraphx::parse_onnx("gather_elements_axis0_test.onnx");
    p.compile(migraphx::make_target("ref"));
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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(gemm_test)
{
    migraphx::program p = migraphx::parse_onnx("gemm_brcst_C_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a_shape{migraphx::shape::float_type, {5, 6}};
    std::vector<float> a_data = {0.26472837, 0.8525864,  0.41929847, 0.14151508, 0.43216065,
                                 0.67468566, 0.42488748, 0.82021785, 0.9782456,  0.5794279,
                                 0.6627283,  0.4790396,  0.9237051,  0.7340607,  0.67379653,
                                 0.87168175, 0.37324256, 0.33278653, 0.42736676, 0.024699844,
                                 0.75851107, 0.48719302, 0.5834426,  0.6938476,  0.43747696,
                                 0.24054702, 0.26912406, 0.6760658,  0.5419149,  0.89949054};

    migraphx::shape b_shape{migraphx::shape::float_type, {5, 7}};
    std::vector<float> b_data = {
        0.65727437,  0.54262096, 0.14126152, 0.8994123,  0.21831702,  0.81191784, 0.9371278,
        0.3438551,   0.7121373,  0.90316695, 0.26614252, 0.80144906,  0.80301756, 0.49930334,
        0.0719704,   0.63484156, 0.7343097,  0.32130218, 0.7094916,   0.6116475,  0.74144083,
        0.021210382, 0.38724765, 0.44830495, 0.62347615, 0.022489505, 0.23316588, 0.76540905,
        0.895689,    0.81540287, 0.223875,   0.9275573,  0.4621397,   0.70785195, 0.5658555};

    migraphx::shape c_shape{migraphx::shape::float_type, {6, 1}};
    std::vector<float> c_data = {
        0.07358502, 0.13792239, 0.8574055, 0.40553397, 0.38205826, 0.62062204};

    migraphx::parameter_map params;
    params["A"] = migraphx::argument(a_shape, a_data.data());
    params["B"] = migraphx::argument(b_shape, b_data.data());
    params["C"] = migraphx::argument(c_shape, c_data.data());

    auto result = p.eval(params).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        0.45261115, 0.83629227, 0.7533463,  0.7189715, 0.69160205, 0.824082,  0.9187499,
        0.6659525,  0.96956736, 0.84293026, 0.8400868, 0.84835225, 1.0982862, 1.0642393,
        1.1447254,  1.6184721,  1.6048342,  1.4741788, 1.4334437,  1.638659,  1.7428316,
        0.8098607,  1.2157929,  1.1010075,  1.0706307, 1.0429881,  1.1771785, 1.2362702,
        0.8239243,  1.1112559,  0.9639262,  1.0813537, 0.8825792,  1.121141,  1.1885703,
        1.2227502,  1.4568202,  1.1388762,  1.55058,   1.0958102,  1.4637487, 1.5756242};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(gemm_half_test)
{
    migraphx::program p = migraphx::parse_onnx("gemm_half_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a_shape{migraphx::shape::half_type, {8, 6}};
    std::vector<float> tmp = {0.2646, 0.8525, 0.4192, 0.1415, 0.4321,  0.675,  0.4248, 0.8203,
                              0.978,  0.5796, 0.6626, 0.479,  0.924,   0.734,  0.674,  0.8716,
                              0.3733, 0.3328, 0.4272, 0.0247, 0.7583,  0.4873, 0.5835, 0.694,
                              0.4375, 0.2406, 0.269,  0.6763, 0.542,   0.8994, 0.657,  0.5425,
                              0.1412, 0.8994, 0.2183, 0.812,  0.937,   0.3438, 0.712,  0.9033,
                              0.266,  0.8013, 0.803,  0.4993, 0.07196, 0.635,  0.7344, 0.3213};
    std::vector<migraphx::half> a_data{tmp.cbegin(), tmp.cend()};

    migraphx::shape b_shape{migraphx::shape::half_type, {8, 7}};
    tmp = {0.7095,  0.612,  0.741,  0.02121, 0.3872, 0.4482,  0.6235,  0.02249, 0.2332, 0.7656,
           0.8955,  0.8154, 0.2239, 0.9277,  0.4622, 0.708,   0.566,   0.0736,  0.138,  0.8574,
           0.4055,  0.382,  0.6206, 0.424,   0.3674, 0.435,   0.998,   0.3594,  0.701,  0.6216,
           0.01826, 0.6313, 0.514,  0.1095,  0.3203, 0.01636, 0.537,   0.01952, 0.4502, 0.8965,
           0.5415,  0.7456, 0.793,  0.756,   0.9,    0.5264,  0.05368, 0.4214,  0.276,  0.1517,
           0.08453, 0.83,   0.417,  0.1682,  0.845,  0.1729};
    std::vector<migraphx::half> b_data{tmp.cbegin(), tmp.cend()};

    migraphx::shape c_shape{migraphx::shape::half_type, {6, 1}};
    tmp = {0.10846, 0.672, 0.527, 0.94, 0.429, 0.2291};
    std::vector<migraphx::half> c_data{tmp.cbegin(), tmp.cend()};

    migraphx::parameter_map params;
    params["A"] = migraphx::argument(a_shape, a_data.data());
    params["B"] = migraphx::argument(b_shape, b_data.data());
    params["C"] = migraphx::argument(c_shape, c_data.data());

    auto result = p.eval(params).back();
    std::vector<migraphx::half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    tmp = {1.071, 1.378, 1.465, 1.093, 0.968, 1.542, 1.145, 1.287,  1.533, 1.75,  1.338,
           1.449, 1.592, 1.668, 1.265, 1.531, 1.656, 1.348, 1.2705, 1.525, 1.479, 1.754,
           2.143, 2.062, 1.921, 1.836, 2.203, 1.952, 1.055, 1.225,  1.418, 1.209, 1.155,
           1.42,  1.234, 1.302, 1.593, 1.368, 1.289, 1.327, 1.451,  1.394};
    std::vector<migraphx::half> gold{tmp.cbegin(), tmp.cend()};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

template <typename T = float>
std::vector<T> norm_test(const std::vector<size_t>& x_dims,
                         std::vector<T>& scale,
                         std::vector<T>& bias,
                         const std::string& onnx_file)
{
    migraphx::program p = migraphx::parse_onnx(onnx_file);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s_x{migraphx::shape::get_type<T>{}, x_dims};
    migraphx::shape s_s{migraphx::shape::get_type<T>{}, {scale.size()}};
    migraphx::shape s_b{migraphx::shape::get_type<T>{}, {scale.size()}};

    std::vector<T> x(s_x.elements());
    std::iota(std::begin(x), std::end(x), 1);

    migraphx::parameter_map pp;
    pp["x"]     = migraphx::argument(s_x, x.data());
    pp["scale"] = migraphx::argument(s_s, scale.data());
    pp["bias"]  = migraphx::argument(s_b, bias.data());

    auto result = p.eval(pp).back();

    std::vector<T> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    return result_vector;
}

TEST_CASE(group_norm_test)
{
    std::vector<float> scale{1.2, 0.8};
    std::vector<float> bias{0.5, 0.2};
    std::vector<float> result_vector =
        norm_test<float>({1, 4, 2}, scale, bias, "group_norm_3d_test.onnx");
    std::vector<float> gold = {-1.10996256,
                               -0.0366542,
                               1.0366542,
                               2.10996256,
                               -0.87330837,
                               -0.15776947,
                               0.55776947,
                               1.27330837};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(group_norm_half_test)
{
    using migraphx::half;
    std::vector<half> scale{half{1.2}, half{0.8}};
    std::vector<half> bias{half{0.5}, half{0.2}};
    std::vector<half> result_vector =
        norm_test<half>({1, 4, 2}, scale, bias, "group_norm_3d_half_test.onnx");
    std::vector<half> gold = {half{-1.10996256},
                              half{-0.0366542},
                              half{1.0366542},
                              half{2.10996256},
                              half{-0.87330837},
                              half{-0.15776947},
                              half{0.55776947},
                              half{1.27330837}};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(greaterorequal_test)
{
    migraphx::program p = migraphx::parse_onnx("greaterorequal_test.onnx");
    p.compile(migraphx::make_target("ref"));

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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(hardsigmoid_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("hardsigmoid_verify_test.onnx");
    p.compile(migraphx::make_target("ref"));

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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(if_else_test)
{
    migraphx::program p = migraphx::parse_onnx("if_else_test.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape s_data{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {0.0625, 0.75, -0.0625, 0.125, -0.125, -0.5625};
    migraphx::shape bool_data{migraphx::shape::bool_type, {1}};
    bool b_data = false;

    migraphx::parameter_map pp;
    pp["x"]    = migraphx::argument(s_data, data.data());
    pp["y"]    = migraphx::argument(s_data, data.data());
    pp["cond"] = migraphx::argument(bool_data, &b_data);

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.0866565, -0.371067, 0.017719, 0.0250614, 0.0612539, -0.744683};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(if_else_test_inlined)
{
    migraphx::program p = migraphx::parse_onnx("if_else_test_inlined.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape s_data{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {0.0625, 0.75, -0.0625, 0.125, -0.125, -0.5625};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_data, data.data());
    pp["y"] = migraphx::argument(s_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.0507132, -0.712328, 0.0105797, 0.04569, 0.0185013, -1.16472};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(if_then_test)
{
    migraphx::program p = migraphx::parse_onnx("if_then_test.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape s_data{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {0.0625, 0.75, -0.0625, 0.125, -0.125, -0.5625};
    migraphx::shape bool_data{migraphx::shape::bool_type, {1}};
    bool b_data = true;

    migraphx::parameter_map pp;
    pp["x"]    = migraphx::argument(s_data, data.data());
    pp["y"]    = migraphx::argument(s_data, data.data());
    pp["cond"] = migraphx::argument(bool_data, &b_data);

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // onnx adds ones so result should be just + 1.0
    std::vector<float> gold = {1.0625, 1.75, 0.9375, 1.125, 0.875, 0.4375};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(if_then_test_inlined)
{
    migraphx::program p = migraphx::parse_onnx("if_then_test_inlined.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape s_data{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {0.0625, 0.75, -0.0625, 0.125, -0.125, -0.5625};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_data, data.data());
    pp["y"] = migraphx::argument(s_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1.0625, 1.75, 0.9375, 1.125, 0.875, 0.4375};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(if_literal_test)
{
    auto run_prog = [](bool cond) {
        migraphx::program p = migraphx::parse_onnx("if_literal_test.onnx");
        p.compile(migraphx::make_target("ref"));
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
        EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
    }

    // else branch
    {
        auto result_vector      = run_prog(false);
        std::vector<float> gold = {5, 4, 3, 2, 1};
        EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
    }
}

TEST_CASE(if_then_else_multi_output_shapes_inlined_test)
{
    migraphx::program p =
        migraphx::parse_onnx("if_then_else_multi_output_shapes_inlined_test.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape x_data{migraphx::shape::float_type, {2, 3, 1}};
    migraphx::shape y_data{migraphx::shape::float_type, {2, 3}};
    std::vector<float> data = {0.0625, 0.75, -0.0625, 0.125, -0.125, -0.5625};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(x_data, data.data());
    pp["y"] = migraphx::argument(y_data, data.data());

    auto result_args = p.eval(pp);
    auto result      = result_args.front();
    auto result_b    = result_args.back();

    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> result_vector_back;
    result_b.visit([&](auto output) { result_vector_back.assign(output.begin(), output.end()); });

    result_vector.insert(result_vector.end(), result_vector_back.begin(), result_vector_back.end());

    std::vector<float> gold = {
        1.0625, 1.75, 0.9375, 1.125, 0.875, 0.4375, 0.125, 1.50, -0.125, 0.250, -0.250, -1.125};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(if_then_else_multi_output_shapes_test)
{
    migraphx::program p = migraphx::parse_onnx("if_then_else_multi_output_shapes_test.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape s_data{migraphx::shape::float_type, {2, 3, 1}};
    std::vector<float> data = {0.0625, 0.75, -0.0625, 0.125, -0.125, -0.5625};
    migraphx::shape bool_data{migraphx::shape::bool_type, {1}};
    bool b_data = true;

    migraphx::parameter_map pp;
    pp["x"]    = migraphx::argument(s_data, data.data());
    pp["y"]    = migraphx::argument(s_data, data.data());
    pp["cond"] = migraphx::argument(bool_data, &b_data);

    auto result_args = p.eval(pp);
    auto result      = result_args.front();
    auto result_b    = result_args.back();

    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> result_vector_back;
    result_b.visit([&](auto output) { result_vector_back.assign(output.begin(), output.end()); });

    result_vector.insert(result_vector.end(), result_vector_back.begin(), result_vector_back.end());

    std::vector<float> gold = {
        1.0625, 1.75, 0.9375, 1.125, 0.875, 0.4375, 0.125, 1.50, -0.125, 0.250, -0.250, -1.125};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(if_pl_test)
{
    auto run_prog = [](bool cond) {
        migraphx::program p = migraphx::parse_onnx("if_pl_test.onnx");
        p.compile(migraphx::make_target("ref"));
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
        EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
    }

    // else branch
    {
        auto result_vector      = run_prog(false);
        std::vector<float> gold = {1, 2, 3, 4, 5, 6};
        EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
    }
}

TEST_CASE(if_tuple_test)
{
    auto run_prog = [](bool cond) {
        migraphx::program p = migraphx::parse_onnx("if_tuple_test.onnx");
        p.compile(migraphx::make_target("ref"));
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
        EXPECT(migraphx::verify::verify_rms_range(results.at(0), gold0));
        EXPECT(migraphx::verify::verify_rms_range(results.at(1), gold1));
    }

    // else branch
    {
        auto results = run_prog(false);
        std::vector<float> gold0(4, 3.0f);
        std::vector<float> gold1(12, 5.0f);
        EXPECT(migraphx::verify::verify_rms_range(results.at(0), gold0));
        EXPECT(migraphx::verify::verify_rms_range(results.at(1), gold1));
    }
}

TEST_CASE(instance_norm_test)
{
    migraphx::program p = migraphx::parse_onnx("instance_norm_val_test.onnx");

    p.compile(migraphx::make_target("ref"));
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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(instance_norm_dyn_batch_test)
{
    migraphx::program p = migraphx::parse_onnx("instance_norm_dyn_batch_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s0{migraphx::shape::float_type, {1, 2, 3, 3}};
    std::vector<float> data0 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8};
    migraphx::shape s1{migraphx::shape::float_type, {2}};
    std::vector<float> data1 = {1, 2};
    migraphx::shape s2{migraphx::shape::float_type, {2}};
    std::vector<float> data2 = {0, 1};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(s0, data0.data());
    pp["1"] = migraphx::argument(s1, data1.data());
    pp["2"] = migraphx::argument(s2, data2.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(instance_norm_3d_test)
{
    migraphx::program p = migraphx::parse_onnx("instance_norm_val_3d_test.onnx");

    p.compile(migraphx::make_target("ref"));
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

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(isinf_half_test)
{
    migraphx::program p = migraphx::parse_onnx("isinf_half_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::half_type, {2, 3}};
    migraphx::parameter_map pp;
    migraphx::half nan               = std::numeric_limits<migraphx::half>::quiet_NaN();
    migraphx::half infinity          = std::numeric_limits<migraphx::half>::infinity();
    migraphx::half max               = std::numeric_limits<migraphx::half>::max();
    migraphx::half min               = std::numeric_limits<migraphx::half>::min();
    migraphx::half val               = migraphx::half(3.6);
    std::vector<migraphx::half> data = {-infinity, nan, min, val, max, infinity};
    pp["t1"]                         = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1, 0, 0, 0, 0, 1};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(isinf_neg_test)
{
    migraphx::program p = migraphx::parse_onnx("isinf_neg_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::parameter_map pp;
    float nan               = std::numeric_limits<float>::quiet_NaN();
    float infinity          = std::numeric_limits<float>::infinity();
    float max               = std::numeric_limits<float>::max();
    float min               = std::numeric_limits<float>::min();
    std::vector<float> data = {-infinity, nan, min, 3.6, max, infinity};
    pp["t1"]                = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(isinf_double_pos_test)
{
    migraphx::program p = migraphx::parse_onnx("isinf_double_pos_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::double_type, {2, 3}};
    migraphx::parameter_map pp;
    double nan               = std::numeric_limits<double>::quiet_NaN();
    double infinity          = std::numeric_limits<double>::infinity();
    double max               = std::numeric_limits<double>::max();
    double min               = std::numeric_limits<double>::min();
    std::vector<double> data = {-infinity, nan, min, 3.6, max, infinity};
    pp["t1"]                 = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0, 0, 0, 0, 0, 1};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(isinf_no_detect_test)
{
    migraphx::program p = migraphx::parse_onnx("isinf_no_detect_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::parameter_map pp;
    float nan                = std::numeric_limits<float>::quiet_NaN();
    float infinity           = std::numeric_limits<float>::infinity();
    float max                = std::numeric_limits<float>::max();
    float min                = std::numeric_limits<float>::min();
    std::vector<double> data = {-infinity, nan, min, 3.6, max, infinity};
    pp["t1"]                 = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0, 0, 0, 0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(layer_norm_test)
{
    std::vector<float> scale{1.2, 0.8};
    std::vector<float> bias{0.5, 0.2};
    std::vector<float> result_vector =
        norm_test<float>({1, 4, 2}, scale, bias, "layer_norm_3d_test.onnx");
    std::vector<float> gold = {-0.69997597,
                               0.99998398,
                               -0.69997597,
                               0.99998398,
                               -0.69997597,
                               0.99998398,
                               -0.69997597,
                               0.99998398};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(layer_norm_half_test)
{
    using migraphx::half;
    std::vector<half> scale{half{1.2}, half{0.8}};
    std::vector<half> bias{half{0.5}, half{0.2}};
    std::vector<half> result_vector =
        norm_test<half>({1, 4, 2}, scale, bias, "layer_norm_3d_half_test.onnx");
    std::vector<half> gold = {half{-0.69997597},
                              half{0.99998398},
                              half{-0.69997597},
                              half{0.99998398},
                              half{-0.69997597},
                              half{0.99998398},
                              half{-0.69997597},
                              half{0.99998398}};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(lessorequal_test)
{
    migraphx::program p = migraphx::parse_onnx("lessorequal_test.onnx");
    p.compile(migraphx::make_target("ref"));

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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(lpnormalization_1norm)
{
    migraphx::program p = migraphx::parse_onnx("lpnormalization_l1_test.onnx");
    p.compile(migraphx::make_target("ref"));
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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(lpnormalization_2norm)
{
    migraphx::program p = migraphx::parse_onnx("lpnormalization_l2_test.onnx");
    p.compile(migraphx::make_target("ref"));
    migraphx::shape s{migraphx::shape::float_type, {3, 4}};
    std::vector<float> data{0.f, 2.f, -2.f, 1.f, 1.f, -5.f, 3.f, -1.f, -4.f, 3.f, 0.f, 0.f};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold{0.f,
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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(mean_broadcast_test)
{
    migraphx::program p = migraphx::parse_onnx("mean_broadcast_test.onnx");
    p.compile(migraphx::make_target("ref"));

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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(mean_test)
{
    migraphx::program p = migraphx::parse_onnx("mean_test.onnx");
    p.compile(migraphx::make_target("ref"));

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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(mean_integral_test)
{
    migraphx::program p = migraphx::parse_onnx("mean_integral_test.onnx");
    p.compile(migraphx::make_target("ref"));

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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

template <typename T = float>
std::vector<T> mvn_test(std::vector<size_t> data_lens, const std::string& test_file)
{
    migraphx::program p = migraphx::parse_onnx(test_file);
    p.compile(migraphx::make_target("ref"));

    migraphx::shape data_shape(migraphx::shape::get_type<T>{}, std::move(data_lens));
    std::vector<T> data(data_shape.elements());
    std::iota(begin(data), end(data), 0);

    migraphx::parameter_map pm;
    pm["data"] = migraphx::argument(data_shape, data.data());

    auto result = p.eval(pm).back();
    std::vector<T> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    return result_vector;
}

TEST_CASE(mvn_default_axes_test)
{
    auto result = mvn_test({2, 2, 2, 2}, "mvn_default_axes_test.onnx");
    std::vector<float> gold{-1.32424438,
                            -1.08347268,
                            -0.84270097,
                            -0.60192927,
                            -1.32424438,
                            -1.08347268,
                            -0.84270097,
                            -0.60192927,
                            0.60192927,
                            0.84270097,
                            1.08347268,
                            1.32424438,
                            0.60192927,
                            0.84270097,
                            1.08347268,
                            1.32424438};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}

TEST_CASE(mvn_default_axes_fp16_test)
{
    using migraphx::half;
    auto result = mvn_test<half>({2, 2, 2, 2}, "mvn_default_axes_fp16_test.onnx");
    std::vector<half> gold{half{-1.324},
                           half{-1.084},
                           half{-0.843},
                           half{-0.602},
                           half{-1.324},
                           half{-1.084},
                           half{-0.843},
                           half{-0.602},
                           half{0.602},
                           half{0.843},
                           half{1.084},
                           half{1.324},
                           half{0.602},
                           half{0.843},
                           half{1.084},
                           half{1.324}};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}

TEST_CASE(mvn_rank_2_test)
{
    auto result = mvn_test({2, 2}, "mvn_rank_2_test.onnx");
    std::vector<float> gold{-1, 1, -1, 1};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}

TEST_CASE(mvn_rank_2_fp16_test)
{
    using migraphx::half;
    auto result = mvn_test<migraphx::half>({2, 2}, "mvn_rank_2_fp16_test.onnx");
    std::vector<migraphx::half> gold{half{-1}, half{1}, half{-1}, half{1}};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}

TEST_CASE(mvn_rank_3_test)
{
    auto result = mvn_test({2, 2, 2}, "mvn_rank_3_test.onnx");
    std::vector<float> gold{-1.34164079,
                            -1.34164079,
                            -0.4472136,
                            -0.4472136,
                            0.4472136,
                            0.4472136,
                            1.34164079,
                            1.34164079};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}

TEST_CASE(mvn_rank_3_fp16_test)
{
    using migraphx::half;
    auto result = mvn_test<half>({2, 2, 2}, "mvn_rank_3_fp16_test.onnx");
    std::vector<half> gold{half{-1.342},
                           half{-1.342},
                           half{-0.4473},
                           half{-0.4473},
                           half{0.4473},
                           half{0.4473},
                           half{1.342},
                           half{1.342}};
    EXPECT(migraphx::verify::verify_rms_range(result, gold));
}

TEST_CASE(mod_test)
{
    migraphx::program p = migraphx::parse_onnx("mod_test.onnx");
    p.compile(migraphx::make_target("ref"));

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

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(mod_test_different_types)
{
    migraphx::program p = migraphx::parse_onnx("mod_test_different_dtypes.onnx");
    p.compile(migraphx::make_target("ref"));

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

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(mod_test_fmod)
{
    migraphx::program p = migraphx::parse_onnx("mod_test_fmod.onnx");
    p.compile(migraphx::make_target("ref"));

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

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(mod_test_fmod_different_types)
{
    migraphx::program p = migraphx::parse_onnx("mod_test_fmod_different_dtypes.onnx");
    p.compile(migraphx::make_target("ref"));

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

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(multinomial_dyn_test)
{
    migraphx::onnx_options options;
    options.default_dyn_dim_value = {1, 4};
    auto p                        = migraphx::parse_onnx("multinomial_dyn_test.onnx", options);
    const size_t batch_size(2);
    const size_t categories(5);
    const size_t sample_size(100000);
    p.compile(migraphx::make_target("ref"));

    // Distribution function (2 distributions of 5 categories each)
    std::vector<int> dist{15, 25, 15, 25, 20, 20, 20, 10, 25, 25};
    EXPECT(dist.size() == categories * batch_size);
    std::vector<float> data(categories * batch_size);

    std::transform(dist.begin(), dist.end(), data.begin(), [&](auto d) { return log(d); });
    // Shape of the probability distribution, which also defines the number of categories
    migraphx::shape s{migraphx::shape::float_type, {batch_size, categories}};

    migraphx::parameter_map pp;
    pp["input"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();

    std::vector<int32_t> result_vec(batch_size * sample_size);
    result.visit([&](auto output) { result_vec.assign(output.begin(), output.end()); });

    // Make a categorical histogram of output
    // for first result in batch
    std::vector<int> res_dist(categories, 0);
    size_t r = 0;
    for(r = 0; r < result_vec.size() / 2; r++)
        res_dist[result_vec[r]]++;

    // normalizing factors for original and measured distributions
    auto dist_sum     = std::accumulate(dist.begin(), dist.begin() + 5, 0);
    auto res_dist_sum = std::accumulate(res_dist.begin(), res_dist.end(), 0);

    //  Values approximate the distribution in dist
    std::vector<float> norm(5);
    std::vector<float> res_norm(5);

    std::transform(dist.begin(), dist.begin() + 5, norm.begin(), [&](auto n) {
        return static_cast<double>(n) / dist_sum;
    });
    std::transform(res_dist.begin(), res_dist.end(), res_norm.begin(), [&](auto n) {
        return static_cast<double>(n) / res_dist_sum;
    });

    EXPECT(migraphx::verify::verify_range_with_tolerance(
        norm, migraphx::verify::expected{res_norm}, migraphx::verify::tolerance{0.01}));

    // Make a categorical histogram of output
    // for second result in batch
    std::fill(res_dist.begin(), res_dist.end(), 0);
    for(; r < result_vec.size(); r++)
        res_dist[result_vec[r]]++;

    dist_sum     = std::accumulate(dist.begin() + 5, dist.end(), 0);
    res_dist_sum = std::accumulate(res_dist.begin(), res_dist.end(), 0);
    std::transform(dist.begin() + 5, dist.end(), norm.begin(), [&](auto n) {
        return static_cast<double>(n) / dist_sum;
    });
    std::transform(res_dist.begin(), res_dist.end(), res_norm.begin(), [&](auto n) {
        return static_cast<double>(n) / res_dist_sum;
    });

    EXPECT(migraphx::verify::verify_range_with_tolerance(
        res_norm, migraphx::verify::expected{norm}, migraphx::verify::tolerance{0.01}));
}

TEST_CASE(nonzero_test)
{
    migraphx::program p = migraphx::parse_onnx("nonzero_dynamic_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::bool_type, {2, 2}};
    std::vector<char> data = {1, 1, 1, 0};

    migraphx::parameter_map pp;
    pp["data"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0, 0, 1, 0, 0, 1, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearadd_test)
{
    // github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearAdd
    migraphx::program p = migraphx::parse_onnx("qlinearadd_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a{migraphx::shape::uint8_type, {64}};
    std::vector<uint8_t> data_a = {0,   2,   4,   6,   8,   10,  12,  14,  16,  18,  20,  22,  24,
                                   26,  28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,
                                   52,  54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,
                                   78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98,  100, 102,
                                   104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126};

    migraphx::shape b{migraphx::shape::uint8_type, {64}};
    std::vector<uint8_t> data_b = {128, 126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104,
                                   102, 100, 98,  96,  94,  92,  90,  88,  86,  84,  82,  80,  78,
                                   76,  74,  72,  70,  68,  66,  64,  62,  60,  58,  56,  54,  52,
                                   50,  48,  46,  44,  42,  40,  38,  36,  34,  32,  30,  28,  26,
                                   24,  22,  20,  18,  16,  14,  12,  10,  8,   6,   4,   2};

    migraphx::parameter_map pp;
    pp["A"]     = migraphx::argument(a, data_a.data());
    pp["B"]     = migraphx::argument(b, data_b.data());
    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
                                 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
                                 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64,
                                 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearadd_bcast_test)
{
    // github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearAdd
    migraphx::program p = migraphx::parse_onnx("qlinearadd_bcast_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a{migraphx::shape::int8_type, {64}};
    std::vector<int8_t> data_a = {-64, -62, -60, -58, -56, -54, -52, -50, -48, -46, -44, -42, -40,
                                  -38, -36, -34, -32, -30, -28, -26, -24, -22, -20, -18, -16, -14,
                                  -12, -10, -8,  -6,  -4,  -2,  0,   2,   4,   6,   8,   10,  12,
                                  14,  16,  18,  20,  22,  24,  26,  28,  30,  32,  34,  36,  38,
                                  40,  42,  44,  46,  48,  50,  52,  54,  56,  58,  60,  62};

    migraphx::shape b{migraphx::shape::int8_type, {1, 1, 64}};
    std::vector<int8_t> data_b = {96, 94,  92,  90,  88,  86,  84,  82,  80,  78,  76,  74, 72,
                                  70, 68,  66,  64,  62,  60,  58,  56,  54,  52,  50,  48, 46,
                                  44, 42,  40,  38,  36,  34,  32,  30,  28,  26,  24,  22, 20,
                                  18, 16,  14,  12,  10,  8,   6,   4,   2,   0,   -2,  -4, -6,
                                  -8, -10, -12, -14, -16, -18, -20, -22, -24, -26, -28, -30};

    migraphx::parameter_map pp;
    pp["A"]     = migraphx::argument(a, data_a.data());
    pp["B"]     = migraphx::argument(b, data_b.data());
    auto result = p.eval(pp).back();

    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {-64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64,
                                -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64,
                                -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64,
                                -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64,
                                -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64, -64};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearaveragepool_1d_test)
{
    auto p = migraphx::parse_onnx("qlinearaveragepool_1d_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<int8_t> data_x = {
        -31,  51,  125,  30,   -17,  -125, 121,  -19, -13,  52,   18,  -70,  97,   15,  56,   42,
        -65,  -26, 40,   -109, -70,  83,   110,  -94, 34,   70,   5,   -23,  -60,  -68, 19,   48,
        -113, 3,   -44,  20,   -99,  -103, -49,  -38, 122,  75,   38,  -7,   -65,  -56, 96,   99,
        50,   -27, -114, 49,   -65,  105,  -3,   54,  8,    38,   -81, -46,  -86,  -46, -104, 36,
        22,   -51, 48,   59,   -116, 6,    93,   16,  -111, 98,   51,  -87,  -111, -74, -39,  7,
        107,  115, 59,   60,   -66,  -14,  -106, -23, 119,  -122, -51, -100, 26,   125, 45,   90};
    migraphx::shape s_x{migraphx::shape::int8_type, {1, 3, 32}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {
        26,  104, 94,  22,  -55, 14,  67, 0,   36,  51,  -10, 29,  72,  52,  65,  5,
        -30, 23,  -19, -74, 23,  112, 24, -14, 68,  54,  7,   -26, -48, -8,  50,  -39,
        -4,  4,   -24, -85, -60, -28, 58, 114, 72,  31,  -20, -44, 36,  114, 90,  28,
        -54, -16, 8,   36,  67,  42,  47, 39,  -6,  -48, -50, -50, -59, -18, 2,   15,
        70,  -13, -39, 66,  71,  -32, 9,  90,  -2,  -83, -76, -40, 0,   73,  127, 103,
        75,  13,  -24, -44, -48, 64,  15, -70, -60, -21, 92,  101, 84};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearaveragepool_2d_test)
{
    auto p = migraphx::parse_onnx("qlinearaveragepool_2d_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<int8_t> data_x = {84,  -73, 117, -2,  -97, 72,   67,  27,   1,  -44,  110, 51,
                                  9,   7,   58,  113, -34, 34,   124, -20,  6,  66,   68,  98,
                                  31,  -84, 25,  101, -69, -100, -68, 116,  33, -121, 78,  49,
                                  102, -86, 65,  69,  -87, -89,  16,  -125, 51, -54,  -86, 79};
    migraphx::shape s_x{migraphx::shape::int8_type, {1, 3, 4, 4}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {4,   127, 127, -41,  127, 127, -6,   125,  127,
                                76,  127, 127, 32,   78,  127, -128, -128, 127,
                                -44, -37, 127, -117, -62, 37,  -128, -128, -81};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearaveragepool_2d_ceil_test)
{
    auto p = migraphx::parse_onnx("qlinearaveragepool_2d_ceil_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<uint8_t> data_x = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};
    migraphx::shape s_x{migraphx::shape::uint8_type, {1, 1, 4, 4}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {120, 150, 240, 255};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearaveragepool_2d_dilations_test)
{
    auto p = migraphx::parse_onnx("qlinearaveragepool_2d_dilations_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<int8_t> data_x = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32};
    migraphx::shape s_x{migraphx::shape::int8_type, {1, 1, 4, 4}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {108, 112, 124, 127};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearaveragepool_2d_pads_count_include_pad_test)
{
    auto p = migraphx::parse_onnx("qlinearaveragepool_2d_pads_count_include_pad_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<int8_t> data_x = {-30,  50,  91,  -87,  -21, -113, -16, 6,    -128, 104,  82,  -126,
                                  54,   41,  -71, 62,   -11, -111, 13,  104,  -43,  -48,  30,  85,
                                  -62,  -33, -27, -114, 32,  -17,  30,  -26,  -18,  15,   17,  100,
                                  -122, 115, 84,  -34,  -86, 82,   102, -117, -91,  -105, 112, 91};
    migraphx::shape s_x{migraphx::shape::int8_type, {1, 3, 4, 4}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {
        15,   43,  94,  62,  34,  -16, 4,   -31, 10,  -6,  29,  -13, -67, -45,  43,   27,  4,   -83,
        -21,  -3,  -6,  15,  -3,  0,   -9,  71,  78,  83,  3,   -4,  62,  85,   45,   50,  27,  66,
        26,   -36, -29, 35,  97,  90,  2,   -86, -62, 73,  127, 127, -32, -128, -128, -24, 83,  74,
        -9,   -63, -45, -35, 20,  1,   15,  -12, -11, -72, -44, -46, 50,  40,   57,   25,  34,  18,
        22,   30,  40,  105, 97,  88,  -46, 26,  83,  127, 125, 69,  -94, 24,   127,  127, 116, 4,
        -128, -83, 83,  127, 127, -1,  -66, -79, 40,  124, 127, 18,  -19, -77,  -15,  86,  127, 83};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearaveragepool_2d_same_lower_test)
{
    auto p = migraphx::parse_onnx("qlinearaveragepool_2d_same_lower_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<uint8_t> data_x = {195, 102, 250, 61,  222, 6,   243, 218, 230, 105, 36,  116,
                                   194, 31,  113, 85,  126, 204, 80,  38,  115, 167, 221, 67,
                                   69,  140, 11,  209, 136, 120, 39,  96,  29,  5,   167, 40,
                                   58,  51,  157, 179, 244, 149, 76,  243, 126, 144, 192, 199};
    migraphx::shape s_x{migraphx::shape::uint8_type, {1, 3, 4, 4}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {195, 148, 176, 156, 208, 131, 150, 193, 226, 141, 98,  153,
                                 212, 140, 71,  88,  126, 165, 142, 59,  120, 153, 168, 102,
                                 92,  123, 135, 127, 102, 116, 78,  89,  29,  17,  86,  104,
                                 44,  36,  95,  136, 151, 126, 108, 164, 185, 166, 140, 178};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearaveragepool_2d_same_upper_test)
{
    auto p = migraphx::parse_onnx("qlinearaveragepool_2d_same_upper_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<int8_t> data_x = {-61, 102,  -6,  61,  -34,  6,    -13, -38, -26, 105,  36,  116,
                                  -62, 31,   113, 85,  126,  -52,  80,  38,  115, -89,  -35, 67,
                                  69,  -116, 11,  -47, -120, 120,  39,  96,  29,  5,    -89, 40,
                                  58,  51,   -99, -77, -12,  -107, 76,  -13, 126, -112, -64, -57};
    migraphx::shape s_x{migraphx::shape::int8_type, {1, 3, 4, 4}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {
        -58, -20,  -62,  -41,  -38, 3,    -14,  14,   -40,  78,   111, 127,  -95, 80,   127,  106,
        -14, -112, 11,   41,   -74, -128, -66,  -44,  -88,  -37,  -14, -15,  -64, 95,   71,   127,
        8,   -128, -128, -101, -69, -104, -120, -128, -116, -128, -93, -128, -50, -128, -128, -128};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearaveragepool_2d_strides_test)
{
    auto p = migraphx::parse_onnx("qlinearaveragepool_2d_strides_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<int8_t> data_x = {
        84,   -73,  117, -2,   -97,  72,  67,   27,  1,   -44,  110, 51,   9,    7,    58,  113,
        -34,  34,   124, -20,  6,    66,  68,   98,  31,  -84,  25,  101,  -69,  -100, -68, 116,
        33,   -121, 78,  49,   102,  -86, 65,   69,  -87, -89,  16,  -125, 51,   -54,  -86, 79,
        -112, -37,  -6,  74,   118,  -75, -41,  52,  101, -22,  -28, -92,  -59,  -128, 32,  78,
        -20,  121,  11,  -107, -92,  -31, 81,   117, -55, -3,   80,  119,  126,  -98,  -11, 52,
        -4,   -66,  37,  -57,  -16,  -33, -12,  100, 55,  2,    27,  62,   -15,  64,   -74, -21,
        -123, 22,   -45, 12,   30,   24,  20,   120, -36, -102, -75, -39,  -76,  55,   74,  -120,
        103,  67,   -80, -89,  -112, 36,  69,   98,  110, -82,  60,  119,  98,   88,   5,   42,
        -88,  -86,  -58, -33,  93,   80,  -57,  -56, 87,  7,    -4,  114,  -73,  -91,  -12, -123,
        96,   -99,  -31, -99,  85,   34,  -126, 106, 88,  126,  -60, 14,   75,   -117, -15, 6,
        55,   -14,  117, -87,  -75,  -50, -85,  54,  70,  125,  74,  -100, 25,   -112, 74,  -66,
        -116, -102, 1,   -75,  -107, 83,  -120, -66, 57,  29,   62,  -45,  -103, -56,  90,  -53};
    migraphx::shape s_x{migraphx::shape::int8_type, {1, 3, 8, 8}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {24, 37, 10, 17, 12, 12, -13, -1, 14, -10, 7, -19};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearaveragepool_3d_test)
{
    auto p = migraphx::parse_onnx("qlinearaveragepool_3d_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<int8_t> data_x = {
        -61, 102, -6,  61,  -34, 6,   -13, -38,  -26,  105, 36,  116,  -62, 31,  113,  85,  126,
        -52, 80,  38,  115, -89, -35, 67,  69,   -116, 11,  -47, -120, 120, 39,  96,   29,  5,
        -89, 40,  58,  51,  -99, -77, -12, -107, 76,   -13, 126, -112, -64, -57, 99,   -54, 27,
        99,  126, -46, -7,  109, 17,  77,  94,   -92,  84,  -92, 48,   71,  45,  -102, 95,  118,
        24,  13,  -70, 33,  35,  -60, 102, 81,   34,   108, -79, 14,   -42};
    migraphx::shape s_x{migraphx::shape::int8_type, {1, 3, 3, 3, 3}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {56,  114, 49, 39, 32,  127, 3,   45, -4,  -13, 8,  22,
                                -35, -98, 76, 15, 127, 67,  100, 20, 127, 84,  64, 68};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearaveragepool_notset_test)
{
    auto p = migraphx::parse_onnx("qlinearaveragepool_notset_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<int8_t> data_x = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                  13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    migraphx::shape s_x{migraphx::shape::int8_type, {1, 1, 5, 5}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {22};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearaveragepool_nt_cip_test)
{
    // github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearAveragePool
    auto p = migraphx::parse_onnx("qlinearaveragepool_nt_cip_test.onnx");
    p.compile(migraphx::make_target("ref"));
    std::vector<uint8_t> data_x = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    migraphx::shape s_x{migraphx::shape::uint8_type, {1, 1, 5, 5}};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s_x, data_x.data());

    auto result = p.eval(pp).back();
    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {18};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearconv_test)
{
    // https://xadupre.github.io/draft/onnx/onnx_doc_folder/onnx__QLinearConv.html
    migraphx::program p = migraphx::parse_onnx("qlinearconv_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::uint8_type, {1, 1, 7, 7}};
    std::vector<uint8_t> x_data = {255, 174, 162, 25,  203, 168, 58,  15,  59,  237, 95,  129, 0,
                                   64,  56,  242, 153, 221, 168, 12,  166, 232, 178, 186, 195, 237,
                                   162, 237, 188, 39,  124, 77,  80,  102, 43,  127, 230, 21,  83,
                                   41,  40,  134, 255, 154, 92,  141, 42,  148, 247};

    migraphx::parameter_map pp;
    pp["X"]     = migraphx::argument(sx, x_data.data());
    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {0,   81,  93,  230, 52,  87,  197, 240, 196, 18,  160, 126, 255,
                                 191, 199, 13,  102, 34,  87,  243, 89,  23,  77,  69,  60,  18,
                                 93,  18,  67,  216, 131, 178, 175, 153, 212, 128, 25,  234, 172,
                                 214, 215, 121, 0,   101, 163, 114, 213, 107, 8};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearconv_pad_0_test)
{
    // https:xadupre.github.io/draft/onnx/onnx_doc_folder/onnx__Conv.html

    migraphx::program p = migraphx::parse_onnx("qlinearconv_pad_0_test.onnx");

    p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::uint8_type, {1, 1, 5, 5}};

    std::vector<uint8_t> x_data = {0,   11,  21,  32,  42,  53,  64,  74,  85,  96,  106, 117, 128,
                                   138, 149, 159, 170, 181, 191, 202, 212, 223, 234, 244, 255};

    migraphx::parameter_map pp;
    pp["X"]     = migraphx::argument(sx, x_data.data());
    auto result = p.eval(pp).back();

    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // # (1, 1, 3, 3) output tensor
    std::vector<int8_t> gold = {-43, -29, -15, 28, 42, 56, 99, 113, 127};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearconv_pad_1_test)
{
    // https:xadupre.github.io/draft/onnx/onnx_doc_folder/onnx__Conv.html
    migraphx::program p = migraphx::parse_onnx("qlinearconv_pad_1_test.onnx");

    p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::uint8_type, {1, 1, 5, 5}};

    std::vector<uint8_t> x_data = {0,   11,  21,  32,  42,  53,  64,  74,  85,  96,  106, 117, 128,
                                   138, 149, 159, 170, 181, 191, 202, 212, 223, 234, 244, 255};

    migraphx::parameter_map pp;
    pp["X"]     = migraphx::argument(sx, x_data.data());
    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // # (1, 1, 5, 5) output tensor
    std::vector<uint8_t> gold = {19,  33,  43,  52,  38,  52,  85,  99,  113, 80,  99,  156, 170,
                                 184, 128, 146, 227, 241, 255, 175, 113, 175, 184, 194, 132};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearconv_scale_1D_test)
{
    // https:xadupre.github.io/draft/onnx/onnx_doc_folder/onnx__Conv.html

    migraphx::program p = migraphx::parse_onnx("qlinearconv_scale_1D_test.onnx");

    p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::uint8_type, {1, 1, 5, 5}};

    std::vector<uint8_t> x_data = {0,   11,  21,  32,  42,  53,  64,  74,  85,  96,  106, 117, 128,
                                   138, 149, 159, 170, 181, 191, 202, 212, 223, 234, 244, 255};

    migraphx::parameter_map pp;
    pp["X"]     = migraphx::argument(sx, x_data.data());
    auto result = p.eval(pp).back();

    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // # (1, 2, 3, 3) output tensor
    std::vector<int8_t> gold = {
        -43, -29, -15, 28, 42, 56, 99, 113, 127, -43, -29, -15, 28, 42, 56, 99, 113, 127};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearglobalavgpool_test)
{
    // github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md
    // #com.microsoft.QLinearGlobalAveragePool

    migraphx::program p = migraphx::parse_onnx("qlinearglobalavgpool_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sh_x{migraphx::shape::uint8_type, {1, 3, 4, 4}};
    std::vector<uint8_t> data_x = {160, 156, 152, 148, 144, 140, 136, 132, 124, 120, 116, 112,
                                   108, 104, 100, 96,  64,  72,  80,  88,  96,  104, 112, 120,
                                   136, 144, 152, 160, 168, 176, 184, 192, 120, 121, 122, 123,
                                   124, 125, 126, 127, 129, 130, 131, 132, 133, 134, 135, 136};

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sh_x, data_x.data());

    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {64, 64, 64};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearleakyrelu_test)
{
    // github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearSigmoid
    migraphx::program p = migraphx::parse_onnx("qlinearleakyrelu_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x{migraphx::shape::int8_type, {64}};
    std::vector<int8_t> data_x = {
        -128, -124, -120, -116, -112, -108, -104, -100, -96, -92, -88, -84, -80, -76, -72, -68,
        -64,  -60,  -56,  -52,  -48,  -44,  -40,  -36,  -32, -28, -24, -20, -16, -12, -8,  -4,
        0,    4,    8,    12,   16,   20,   24,   28,   32,  36,  40,  44,  48,  52,  56,  60,
        64,   68,   72,   76,   80,   84,   88,   92,   96,  100, 104, 108, 112, 116, 120, 124};

    migraphx::parameter_map pp;
    pp["X"]     = migraphx::argument(x, data_x.data());
    auto result = p.eval(pp).back();

    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {
        -128, -126, -122, -118, -113, -109, -104, -100, -96, -91, -87, -82, -78, -74, -69, -65,
        -60,  -56,  -52,  -47,  -43,  -38,  -34,  -30,  -25, -21, -16, -12, -8,  -3,  1,   6,
        10,   14,   18,   22,   26,   30,   34,   38,   42,  46,  50,  54,  58,  62,  66,  70,
        74,   78,   82,   86,   90,   94,   98,   102,  106, 110, 114, 118, 122, 126, 127, 127};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearmatmul_1D_test)
{
    migraphx::program p = migraphx::parse_onnx("qlinearmatmul_1D_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a{migraphx::shape::uint8_type, {8}};
    std::vector<uint8_t> data_a = {2, 4, 6, 8, 10, 12, 14, 16};

    migraphx::shape b{migraphx::shape::uint8_type, {8}};
    std::vector<uint8_t> data_b = {126, 130, 124, 132, 122, 134, 120, 136};

    migraphx::parameter_map pp;
    pp["A"]     = migraphx::argument(a, data_a.data());
    pp["B"]     = migraphx::argument(b, data_b.data());
    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {66};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearmatmul_2D_test)
{
    migraphx::program p = migraphx::parse_onnx("qlinearmatmul_2D_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a{migraphx::shape::uint8_type, {1, 8}};
    std::vector<uint8_t> data_a = {2, 4, 6, 8, 10, 12, 14, 16};

    migraphx::shape b{migraphx::shape::uint8_type, {8, 1}};
    std::vector<uint8_t> data_b = {126, 130, 124, 132, 122, 134, 120, 136};

    migraphx::parameter_map pp;
    pp["A"]     = migraphx::argument(a, data_a.data());
    pp["B"]     = migraphx::argument(b, data_b.data());
    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {66};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearmatmul_3D_test)
{
    // https://xadupre.github.io/draft/onnx/onnx_doc_folder/onnx__QLinearMatMul.html

    migraphx::program p = migraphx::parse_onnx("qlinearmatmul_3D_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a{migraphx::shape::uint8_type, {2, 2, 4}};
    std::vector<uint8_t> data_a = {
        208, 236, 0, 238, 3, 214, 255, 29, 208, 236, 0, 238, 3, 214, 255, 29};

    migraphx::shape b{migraphx::shape::uint8_type, {2, 4, 3}};
    std::vector<uint8_t> data_b = {152, 51, 244, 60, 26, 255, 0, 127, 246, 127, 254, 247,
                                   152, 51, 244, 60, 26, 255, 0, 127, 246, 127, 254, 247};

    migraphx::parameter_map pp;
    pp["A"]     = migraphx::argument(a, data_a.data());
    pp["B"]     = migraphx::argument(b, data_b.data());
    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {168, 115, 255, 1, 66, 151, 168, 115, 255, 1, 66, 151};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearmul_test)
{
    // github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearMul
    migraphx::program p = migraphx::parse_onnx("qlinearmul_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a{migraphx::shape::uint8_type, {64}};
    std::vector<uint8_t> data_a = {0,   2,   4,   6,   8,   10,  12,  14,  16,  18,  20,  22,  24,
                                   26,  28,  30,  32,  34,  36,  38,  40,  42,  44,  46,  48,  50,
                                   52,  54,  56,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,
                                   78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98,  100, 102,
                                   104, 106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126};

    migraphx::shape b{migraphx::shape::uint8_type, {64}};
    std::vector<uint8_t> data_b = {128, 126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104,
                                   102, 100, 98,  96,  94,  92,  90,  88,  86,  84,  82,  80,  78,
                                   76,  74,  72,  70,  68,  66,  64,  62,  60,  58,  56,  54,  52,
                                   50,  48,  46,  44,  42,  40,  38,  36,  34,  32,  30,  28,  26,
                                   24,  22,  20,  18,  16,  14,  12,  10,  8,   6,   4,   2};

    migraphx::parameter_map pp;
    pp["A"]     = migraphx::argument(a, data_a.data());
    pp["B"]     = migraphx::argument(b, data_b.data());
    auto result = p.eval(pp).back();

    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<uint8_t> gold = {100, 111, 122, 132, 142, 151, 160, 169, 177, 185, 192, 199, 206,
                                 212, 218, 223, 228, 233, 237, 241, 244, 247, 250, 252, 254, 255,
                                 255, 255, 255, 255, 255, 255, 254, 252, 250, 247, 244, 241, 237,
                                 233, 228, 223, 218, 212, 206, 199, 192, 185, 177, 169, 160, 151,
                                 142, 132, 122, 111, 100, 89,  77,  65,  52,  39,  26,  12};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearmul_bcast_test)
{
    // github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearMul
    migraphx::program p = migraphx::parse_onnx("qlinearmul_bcast_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape a{migraphx::shape::int8_type, {64}};
    std::vector<int8_t> data_a = {-64, -62, -60, -58, -56, -54, -52, -50, -48, -46, -44, -42, -40,
                                  -38, -36, -34, -32, -30, -28, -26, -24, -22, -20, -18, -16, -14,
                                  -12, -10, -8,  -6,  -4,  -2,  0,   2,   4,   6,   8,   10,  12,
                                  14,  16,  18,  20,  22,  24,  26,  28,  30,  32,  34,  36,  38,
                                  40,  42,  44,  46,  48,  50,  52,  54,  56,  58,  60,  62};

    migraphx::shape b{migraphx::shape::int8_type, {1, 1, 64}};
    std::vector<int8_t> data_b = {96, 94,  92,  90,  88,  86,  84,  82,  80,  78,  76,  74, 72,
                                  70, 68,  66,  64,  62,  60,  58,  56,  54,  52,  50,  48, 46,
                                  44, 42,  40,  38,  36,  34,  32,  30,  28,  26,  24,  22, 20,
                                  18, 16,  14,  12,  10,  8,   6,   4,   2,   0,   -2,  -4, -6,
                                  -8, -10, -12, -14, -16, -18, -20, -22, -24, -26, -28, -30};

    migraphx::parameter_map pp;
    pp["A"]     = migraphx::argument(a, data_a.data());
    pp["B"]     = migraphx::argument(b, data_b.data());
    auto result = p.eval(pp).back();

    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {-128, -128, -128, -128, -128, -128, -128, -128, -128, -126, -118,
                                -109, -101, -93,  -86,  -78,  -70,  -63,  -56,  -49,  -42,  -35,
                                -28,  -21,  -15,  -9,   -2,   4,    10,   15,   21,   27,   32,
                                37,   42,   47,   52,   57,   62,   66,   70,   75,   79,   83,
                                86,   90,   94,   97,   100,  103,  106,  109,  112,  115,  117,
                                119,  122,  124,  126,  127,  127,  127,  127,  127};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(qlinearsigmoid_test)
{
    // github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.QLinearSigmoid
    migraphx::program p = migraphx::parse_onnx("qlinearsigmoid_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape x{migraphx::shape::int8_type, {64}};
    std::vector<int8_t> data_x = {
        -128, -124, -120, -116, -112, -108, -104, -100, -96, -92, -88, -84, -80, -76, -72, -68,
        -64,  -60,  -56,  -52,  -48,  -44,  -40,  -36,  -32, -28, -24, -20, -16, -12, -8,  -4,
        0,    4,    8,    12,   16,   20,   24,   28,   32,  36,  40,  44,  48,  52,  56,  60,
        64,   68,   72,   76,   80,   84,   88,   92,   96,  100, 104, 108, 112, 116, 120, 124};

    migraphx::parameter_map pp;
    pp["X"]     = migraphx::argument(x, data_x.data());
    auto result = p.eval(pp).back();

    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<int8_t> gold = {-128, -127, -127, -127, -127, -127, -126, -126, -126, -125, -125,
                                -124, -123, -122, -120, -119, -117, -114, -112, -108, -104, -99,
                                -94,  -87,  -80,  -71,  -62,  -51,  -39,  -27,  -13,  1,    15,
                                29,   43,   56,   69,   81,   92,   101,  110,  117,  124,  127,
                                127,  127,  127,  127,  127,  127,  127,  127,  127,  127,  127,
                                127,  127,  127,  127,  127,  127,  127,  127,  127};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(resize_downsample_f_test)
{
    migraphx::program p = migraphx::parse_onnx("resize_downsample_f_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 4}};
    std::vector<float> dx(sx.elements());
    std::iota(dx.begin(), dx.end(), 0.0f);

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.0f, 3.0f};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(resize_upsample_linear_ac_test)
{
    migraphx::program p = migraphx::parse_onnx("resize_upsample_linear_ac_test.onnx");
    p.compile(migraphx::make_target("ref"));

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

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(resize_upsample_linear_test)
{
    migraphx::program p = migraphx::parse_onnx("resize_upsample_linear_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    std::vector<float> dx = {1.0f, 2.0f, 3.0f, 4.0f};

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {
        1, 1.25, 1.75, 2, 1.5, 1.75, 2.25, 2.5, 2.5, 2.75, 3.25, 3.5, 3, 3.25, 3.75, 4};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(resize_upsample_pf_test)
{
    migraphx::program p = migraphx::parse_onnx("resize_upsample_pf_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sx{migraphx::shape::float_type, {1, 1, 2, 2}};
    std::vector<float> dx = {1.0f, 2.0f, 3.0f, 4.0f};

    migraphx::parameter_map pp;
    pp["X"] = migraphx::argument(sx, dx.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2,
                               3, 3, 3, 4, 4, 4, 3, 3, 3, 4, 4, 4};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(reversesequence_4D_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("reversesequence_4D_test.onnx");
    p.compile(migraphx::make_target("ref"));

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

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(reversesequence_batch_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("reversesequence_batch_test.onnx");
    p.compile(migraphx::make_target("ref"));

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

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(reversesequence_time_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("reversesequence_time_test.onnx");
    p.compile(migraphx::make_target("ref"));

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

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(round_half_test)
{
    migraphx::program p = migraphx::parse_onnx("round_half_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape xs{migraphx::shape::half_type, {4, 4}};
    std::vector<float> tmp = {-3.51,
                              -3.5,
                              -3.49,
                              -2.51,
                              -2.50,
                              -2.49,
                              -1.6,
                              -1.5,
                              -0.51,
                              -0.5,
                              0.5,
                              0.6,
                              2.4,
                              2.5,
                              3.5,
                              4.5};
    std::vector<migraphx::half> data{tmp.cbegin(), tmp.cend()};
    migraphx::parameter_map param_map;
    param_map["x"] = migraphx::argument(xs, data.data());

    auto result = p.eval(param_map).back();

    std::vector<migraphx::half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    tmp = {-4.0, -4.0, -3.0, -3.0, -2.0, -2.0, -2.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 2.0, 4.0, 4.0};
    std::vector<migraphx::half> gold{tmp.cbegin(), tmp.cend()};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(selu_test)
{
    migraphx::program p = migraphx::parse_onnx("selu_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape xs{migraphx::shape::double_type, {2, 3}};
    std::vector<double> x_data = {1.1, 2.1, 0.0, -1.3, -5.3, 12.0};

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(xs, x_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {0.55, 1.05, 0, -0.10912, -0.149251, 6};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(shrink_hard_test)
{
    migraphx::program p = migraphx::parse_onnx("shrink_hard_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {5}};
    std::vector<float> data{-2, -1, 0, 1, 2};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-2, 0, 0, 0, 2};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(shrink_soft_test)
{
    migraphx::program p = migraphx::parse_onnx("shrink_soft_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::float_type, {5}};
    std::vector<float> data{-2, -1, 0, 1, 2};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {-0.5, 0, 0, 0, 0.5};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(shrink_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("shrink_verify_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::half_type, {5}};
    std::vector<float> tmp = {-10.0, -5.0, 0.0, 5.0, 10.0};
    std::vector<migraphx::half> data{tmp.cbegin(), tmp.cend()};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<migraphx::half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    tmp = {-9.0, -4.0, 1.0, 4.0, 9.0};
    std::vector<migraphx::half> gold{tmp.cbegin(), tmp.cend()};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(shrink_verify2_test)
{
    migraphx::program p = migraphx::parse_onnx("shrink_verify2_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::half_type, {5}};
    std::vector<float> tmp = {-10.0, -5.0, 0.0, 5.0, 10.0};
    std::vector<migraphx::half> data{tmp.cbegin(), tmp.cend()};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<migraphx::half> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    tmp = {-5.0, 0.0, 5.0, 10.0, 5.0};
    std::vector<migraphx::half> gold{tmp.cbegin(), tmp.cend()};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(shrink_int8_test)
{
    migraphx::program p = migraphx::parse_onnx("shrink_int8_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::int8_type, {3, 3}};
    std::vector<int8_t> data{-4, -3, -2, -1, 0, 1, 2, 3, 4};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<int8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<int8_t> gold = {-2, -1, 0, 0, 0, 0, 0, 1, 2};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(shrink_uint8_test)
{
    migraphx::program p = migraphx::parse_onnx("shrink_uint8_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape s{migraphx::shape::uint8_type, {3, 3}};
    std::vector<uint8_t> data{1, 2, 3, 4, 5, 6, 7, 8, 9};
    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());

    auto result = p.eval(pp).back();
    std::vector<uint8_t> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<uint8_t> gold = {0, 0, 0, 0, 0, 10, 11, 12, 13};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(size_verify_test)
{
    migraphx::program p = migraphx::parse_onnx("size_verify_test.onnx");
    p.compile(migraphx::make_target("ref"));

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
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sh_data{migraphx::shape::float_type, {3, 2}};
    std::vector<float> data = {0, 1, 2, 3, 4, 5};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(sh_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {2, 3};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(slice_5arg_test)
{
    migraphx::program p = migraphx::parse_onnx("slice_5arg_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sh_data{migraphx::shape::float_type, {5, 5}}; // start
    std::vector<float> data = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(sh_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {10, 11, 12, 13, 15, 16, 17, 18};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(slice_reverse_test)
{
    migraphx::program p = migraphx::parse_onnx("slice_5arg_reverse_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sh_data{migraphx::shape::float_type, {5, 5}}; // start
    std::vector<float> data = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(sh_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {14, 13, 12, 11, 19, 18, 17, 16};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(slice_step_test)
{
    migraphx::program p = migraphx::parse_onnx("slice_5arg_step_test.onnx");
    p.compile(migraphx::make_target("ref"));

    migraphx::shape sh_data{migraphx::shape::float_type, {5, 5}}; // start
    std::vector<float> data = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

    migraphx::parameter_map pp;
    pp["0"] = migraphx::argument(sh_data, data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    std::vector<float> gold = {14, 12};
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softplus_test)
{
    migraphx::program p = migraphx::parse_onnx("softplus_test.onnx");
    p.compile(migraphx::make_target("ref"));

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

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(softsign_test)
{
    migraphx::program p = migraphx::parse_onnx("softsign_test.onnx");
    p.compile(migraphx::make_target("ref"));

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

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

std::vector<float> gen_trilu_test(const migraphx::shape& s, const migraphx::program& p)
{
    // input data filled with values 1 to nelements
    std::vector<float> x_data(s.elements());
    std::iota(x_data.begin(), x_data.end(), 1);

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, x_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });
    return result_vector;
}

TEST_CASE(triu_test)
{
    migraphx::program p = migraphx::parse_onnx("triu_test.onnx");

    std::vector<float> result_vector = gen_trilu_test({migraphx::shape::float_type, {3, 4}}, p);

    std::vector<float> gold = {1, 2, 3, 4, 0, 6, 7, 8, 0, 0, 11, 12};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(triu_batch_diff_k_test)
{
    migraphx::program p = migraphx::parse_onnx("triu_batch_diff_k_test.onnx");

    std::vector<float> result_vector = gen_trilu_test({migraphx::shape::float_type, {2, 2, 3}}, p);

    std::vector<float> gold = {0, 0, 3, 0, 0, 0, 0, 0, 9, 0, 0, 0};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(tril_test)
{
    migraphx::program p = migraphx::parse_onnx("tril_test.onnx");

    std::vector<float> result_vector = gen_trilu_test({migraphx::shape::float_type, {3, 4}}, p);

    std::vector<float> gold = {1, 0, 0, 0, 5, 6, 0, 0, 9, 10, 11, 0};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(tril_batch_diff_k_test)
{
    migraphx::program p = migraphx::parse_onnx("tril_batch_diff_k_test.onnx");

    std::vector<float> result_vector = gen_trilu_test({migraphx::shape::float_type, {2, 2, 3}}, p);

    std::vector<float> gold = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(triu_neg_k_test)
{
    migraphx::program p = migraphx::parse_onnx("triu_neg_k_test.onnx");

    std::vector<float> result_vector = gen_trilu_test({migraphx::shape::float_type, {3, 4}}, p);

    std::vector<float> gold = {1, 2, 3, 4, 5, 6, 7, 8, 0, 10, 11, 12};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(tril_neg_k_test)
{
    migraphx::program p = migraphx::parse_onnx("tril_neg_k_test.onnx");

    std::vector<float> result_vector = gen_trilu_test({migraphx::shape::float_type, {3, 4}}, p);

    std::vector<float> gold = {0, 0, 0, 0, 5, 0, 0, 0, 9, 10, 0, 0};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(triu_out_k_test)
{
    migraphx::program p = migraphx::parse_onnx("triu_out_k_test.onnx");

    std::vector<float> result_vector = gen_trilu_test({migraphx::shape::float_type, {3, 4}}, p);

    std::vector<float> gold(12, 0);

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(tril_out_k_test)
{
    migraphx::program p = migraphx::parse_onnx("tril_out_k_test.onnx");

    std::vector<float> result_vector = gen_trilu_test({migraphx::shape::float_type, {3, 4}}, p);

    std::vector<float> gold = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(triu_row_one_test)
{
    migraphx::program p = migraphx::parse_onnx("triu_row_one_test.onnx");

    std::vector<float> result_vector = gen_trilu_test({migraphx::shape::float_type, {1, 4}}, p);

    std::vector<float> gold = {0, 2, 3, 4};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(tril_row_one_test)
{
    migraphx::program p = migraphx::parse_onnx("tril_row_one_test.onnx");

    std::vector<float> result_vector = gen_trilu_test({migraphx::shape::float_type, {1, 4}}, p);

    std::vector<float> gold = {1, 2, 0, 0};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

TEST_CASE(unique_dynamic_sorted_test)
{
    migraphx::program p = migraphx::parse_onnx("unique_dynamic_sorted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x{2, 1, 1, 3, 4, 3};
    std::vector<float> y_gold      = {1, 2, 3, 4};
    std::vector<size_t> y_idx_gold = {1, 0, 3, 4};
    std::vector<size_t> x_idx_gold = {1, 0, 0, 2, 3, 2};
    std::vector<size_t> y_ct_gold  = {2, 1, 2, 1};
    migraphx::shape s{migraphx::shape::float_type, {x.size()}};

    migraphx::parameter_map pm;
    pm["X"]     = migraphx::argument(s, x.data());
    auto result = p.eval(pm);

    std::vector<float> yvec;
    result[0].visit([&](auto out) { yvec.assign(out.begin(), out.end()); });
    EXPECT(yvec == y_gold);

    std::vector<size_t> y_idx_vec;
    result[1].visit([&](auto out) { y_idx_vec.assign(out.begin(), out.end()); });
    EXPECT(y_idx_vec == y_idx_gold);

    std::vector<size_t> x_idx_vec;
    result[2].visit([&](auto out) { x_idx_vec.assign(out.begin(), out.end()); });
    EXPECT(x_idx_vec == x_idx_gold);

    std::vector<size_t> y_ct_vec;
    result[3].visit([&](auto out) { y_ct_vec.assign(out.begin(), out.end()); });
    EXPECT(y_ct_vec == y_ct_gold);
}

TEST_CASE(unique_dynamic_unsorted_test)
{
    migraphx::program p = migraphx::parse_onnx("unique_dynamic_unsorted_test.onnx");
    p.compile(migraphx::make_target("ref"));

    std::vector<float> x{2, 1, 1, 3, 4, 3};
    std::vector<float> y_gold      = {2, 1, 3, 4};
    std::vector<size_t> y_idx_gold = {0, 1, 3, 4};
    std::vector<size_t> x_idx_gold = {0, 1, 1, 2, 3, 2};
    std::vector<size_t> y_ct_gold  = {1, 2, 2, 1};
    migraphx::shape s{migraphx::shape::float_type, {x.size()}};

    migraphx::parameter_map pm;
    pm["X"]     = migraphx::argument(s, x.data());
    auto result = p.eval(pm);

    std::vector<float> yvec;
    result[0].visit([&](auto out) { yvec.assign(out.begin(), out.end()); });
    EXPECT(yvec == y_gold);

    std::vector<size_t> y_idx_vec;
    result[1].visit([&](auto out) { y_idx_vec.assign(out.begin(), out.end()); });
    EXPECT(y_idx_vec == y_idx_gold);

    std::vector<size_t> x_idx_vec;
    result[2].visit([&](auto out) { x_idx_vec.assign(out.begin(), out.end()); });
    EXPECT(x_idx_vec == x_idx_gold);

    std::vector<size_t> y_ct_vec;
    result[3].visit([&](auto out) { y_ct_vec.assign(out.begin(), out.end()); });
    EXPECT(y_ct_vec == y_ct_gold);
}

TEST_CASE(where_test)
{
    migraphx::program p = migraphx::parse_onnx("where_test.onnx");
    p.compile(migraphx::make_target("ref"));

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
    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
