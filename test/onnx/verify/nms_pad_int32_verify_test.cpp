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

// NonMaxSuppression selects boxes 3, 0 and 5, and the pad size computed in int32 fills the box
// indices with zeros to 10 entries, the static size the parser gives the output.
TEST_CASE(nms_pad_int32_verify_test)
{
    auto p = read_onnx("nms_pad_int32_test.onnx");
    p.compile(migraphx::make_target("ref"));

    // Center-point format [x_center, y_center, width, height]; boxes 1 and 2 overlap box 0 and
    // box 4 overlaps box 3, so suppression leaves 3 of the 6.
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
    std::vector<float> boxes = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};
    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    migraphx::shape max_out_s{migraphx::shape::int64_type, {1}};
    std::vector<int64_t> max_out = {4};
    migraphx::shape threshold_s{migraphx::shape::float_type, {1}};
    std::vector<float> iou_threshold   = {0.5};
    std::vector<float> score_threshold = {0.0};

    migraphx::parameter_map pp;
    pp["boxes"]                      = migraphx::argument(boxes_s, boxes.data());
    pp["scores"]                     = migraphx::argument(scores_s, scores.data());
    pp["max_output_boxes_per_class"] = migraphx::argument(max_out_s, max_out.data());
    pp["iou_threshold"]              = migraphx::argument(threshold_s, iou_threshold.data());
    pp["score_threshold"]            = migraphx::argument(threshold_s, score_threshold.data());

    auto output = p.eval(pp).back();
    EXPECT(output.get_shape() == migraphx::shape{migraphx::shape::int64_type, {10}});
    std::vector<int64_t> result;
    output.visit([&](auto out) { result.assign(out.begin(), out.end()); });

    std::vector<int64_t> gold = {3, 0, 5, 0, 0, 0, 0, 0, 0, 0};
    EXPECT(result == gold);
}
