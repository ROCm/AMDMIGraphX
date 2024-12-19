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
    std::vector<float> gold = {1.276,     1.546,   1.6359999, 1.9059999, 1.9959998, 2.2659998,

                               13.276001, 13.546,  13.636,    13.906,    13.996,    14.266,

                               25.478498, 26.1535, 25.838501, 26.5135,   26.198502, 26.8735,

                               37.4785,   38.1535, 37.8385,   38.5135,   38.1985,   38.8735};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}

// The half_pixel mode for the ROIAlign op, max pooling
TEST_CASE(roialign_half_pixel_max_verify_test)
{
    migraphx::program p = read_onnx("roialign_half_pixel_max_test.onnx");
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

    // Note:  these Gold values have not been cross-checked with onnxruntime because
    // at the time of writing, onnxruntime is reporting a bug in its own max pooling
    // feature (onnxruntime commit f25f3868a75d4422cde0090abc2781a5277e8eee).  Ref. the
    // following log message in onnxruntime/core/providers/cpu/object_detection/roialign.h:
    //   // TODO(fdwr): Issue #6146. ORT 1.13 will correct the incorrect summation of max mode with
    //   PR #7354. LOGS_DEFAULT(WARNING) << "The existing summation for max mode and sampling ratios
    //   besides 1 is incorrect "
    //                         << "and will be fixed in the next ORT 1.13 release. Thus the results
    //                         of RoiAlign "
    //                         << "will be different.";
    // TODO for AMDMIGraphX:  Recheck the gold values when onnxruntime fix is released
    std::vector<float> gold = {4.700000,  4.700000,  4.700000,  5.280000,  5.280000,  5.280000,
                               15.979999, 15.979999, 15.979999, 13.199999, 13.199999, 13.199999,
                               27.259998, 27.259998, 0.000000,  21.119999, 21.119999, 0.000000,
                               38.539997, 38.539997, 0.000000,  29.039999, 29.039999, 0.000000};

    EXPECT(migraphx::verify::verify_rms_range(result_vector, gold));
}
