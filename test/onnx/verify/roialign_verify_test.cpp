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

    migraphx::shape s{migraphx::shape::float_type, {10, 5, 4, 7}};
    std::vector<float> data(10 * 5 * 4 * 7);
    std::iota(data.begin(), data.end(), 0);

    migraphx::parameter_map pp;
    pp["x"] = migraphx::argument(s, data.data());
    pp["y"] = migraphx::argument(s, data.data());

    migraphx::shape srois{migraphx::shape::float_type, {2, 4}};
    std::vector<float> rois_data = {0.1, 0.15, 0.6, 0.35, 2.1, 1.73, 3.8, 2.13};
    migraphx::shape sbi{migraphx::shape::int64_type, {2}};
    std::vector<int64_t> bi_data = {1, 0};

    pp["rois"]      = migraphx::argument(srois, rois_data.data());
    pp["batch_ind"] = migraphx::argument(sbi, bi_data.data());

    auto result = p.eval(pp).back();
    std::vector<float> result_vector;
    result.visit([&](auto output) { result_vector.assign(output.begin(), output.end()); });

    // gold results were generated with onnxruntime
    std::vector<float> gold = {
        143.16667, 143.49998, 143.83333, 144.56667, 144.9,     145.23334, 145.96667, 146.3,
        146.63333, 147.36667, 147.70001, 148.03334, 148.76666, 149.09999, 149.43333,

        171.16667, 171.5,     171.83333, 172.56667, 172.90001, 173.23334, 173.96667, 174.3,
        174.63333, 175.36667, 175.70001, 176.03333, 176.76666, 177.09999, 177.43335,

        199.16667, 199.5,     199.83333, 200.56667, 200.90001, 201.23334, 201.96666, 202.3,
        202.63333, 203.36665, 203.70001, 204.03333, 204.76668, 205.09999, 205.43333,

        227.16667, 227.5,     227.83333, 228.56668, 228.90001, 229.23332, 229.96669, 230.29999,
        230.63333, 231.36664, 231.70001, 232.03334, 232.76668, 233.09999, 233.43332,

        255.16667, 255.5,     255.83333, 256.56668, 256.90002, 257.2333,  257.96667, 258.3,
        258.63333, 259.36664, 259.69998, 260.03333, 260.7667,  261.09998, 261.43338,

        25.766665, 26.807405, 9.,        25.766665, 26.807405, 9.,        17.177776, 17.871605,
        6.,        0.,        0.,        0.,        0.,        0.,        0.,

        53.766666, 54.807407, 18.333334, 53.766666, 54.807407, 18.333334, 35.844444, 36.538273,
        12.222222, 0.,        0.,        0.,        0.,        0.,        0.,

        81.76667,  82.8074,   27.666666, 81.76667,  82.8074,   27.666666, 54.51111,  55.204937,
        18.444445, 0.,        0.,        0.,        0.,        0.,        0.,

        109.76667, 110.8074,  37.,       109.76667, 110.8074,  37.,       73.17777,  73.871605,
        24.666666, 0.,        0.,        0.,        0.,        0.,        0.,

        137.76666, 138.80742, 46.333332, 137.76666, 138.80742, 46.333332, 91.844444, 92.53828,
        30.88889,  0.,        0.,        0.,        0.,        0.,        0.};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
