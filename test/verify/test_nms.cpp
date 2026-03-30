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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <chrono>
#include <iostream>
#include <thread>

struct test_nms : verify_program<test_nms>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};

        migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
        std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

        auto boxes_l         = mm->add_parameter("boxes", boxes_s);
        auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
        auto max_out_l       = mm->add_literal(int64_t{4});
        auto iou_threshold   = mm->add_literal(0.5f);
        auto score_threshold = mm->add_literal(0.0f);

        auto r =
            mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", 1}}),
                                boxes_l,
                                scores_l,
                                max_out_l,
                                iou_threshold,
                                score_threshold);
        mm->add_return({r});

        return p;
    }
};

// Test NMS with dynamic slice: boxes and scores are sliced by the same
// dynamic end value but have different original spatial dimensions.
// This exercises the overlap-based dynamic dimension check.
struct test_nms_dyn_slice : verify_program<test_nms_dyn_slice>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        // boxes: [1, 10, 4] — 10 box slots
        migraphx::shape boxes_s{migraphx::shape::float_type, {1, 10, 4}};
        std::vector<float> boxes_vec(40);
        for(int i = 0; i < 10; ++i)
        {
            boxes_vec[i * 4 + 0] = i * 0.1f;
            boxes_vec[i * 4 + 1] = i * 0.1f;
            boxes_vec[i * 4 + 2] = i * 0.1f + 0.5f;
            boxes_vec[i * 4 + 3] = i * 0.1f + 0.5f;
        }

        // scores: [1, 1, 5] — only 5 scores
        migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 5}};
        std::vector<float> scores_vec = {0.9f, 0.8f, 0.7f, 0.95f, 0.85f};

        auto boxes_l  = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
        auto scores_l = mm->add_literal(migraphx::literal(scores_s, scores_vec));

        // Dynamic end input for slicing both boxes and scores
        migraphx::shape end_s{migraphx::shape::int64_type, {1}};
        auto end_p = mm->add_parameter("num_valid", end_s);

        auto starts_l = mm->add_literal(migraphx::literal{migraphx::shape::int64_type, {0}});

        // Slice boxes on axis=1: [1,10,4] -> [1,{0..10},4]
        auto sliced_boxes =
            mm->add_instruction(migraphx::make_op("slice", {{"axes", {1}}}),
                                boxes_l,
                                starts_l,
                                end_p);

        // Slice scores on axis=2: [1,1,5] -> [1,1,{0..5}]
        auto sliced_scores =
            mm->add_instruction(migraphx::make_op("slice", {{"axes", {2}}}),
                                scores_l,
                                starts_l,
                                end_p);

        auto max_out_l       = mm->add_literal(int64_t{10});
        auto iou_threshold   = mm->add_literal(0.5f);
        auto score_threshold = mm->add_literal(0.0f);

        auto r = mm->add_instruction(
            migraphx::make_op("nonmaxsuppression", {{"use_dyn_output", true}}),
            sliced_boxes,
            sliced_scores,
            max_out_l,
            iou_threshold,
            score_threshold);
        mm->add_return({r});

        return p;
    }
};

struct test_nms_multi_class : verify_program<test_nms_multi_class>
{
    migraphx::program create_program() const
    {
        std::cout << "test_nms: PID="
            << " waiting 30s for debugger attach..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(30));
        migraphx::program p;
        auto* mm = p.get_main_module();

        // 1 batch, 6 boxes, 2 classes — exercises the corrected
        // max_num_boxes = batches * classes * spatial_dim = 1 * 2 * 6 = 12
        migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
        migraphx::shape scores_s{migraphx::shape::float_type, {1, 2, 6}};
        std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3,
                                         0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

        auto boxes_l         = mm->add_parameter("boxes", boxes_s);
        auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
        auto max_out_l       = mm->add_literal(int64_t{4});
        auto iou_threshold   = mm->add_literal(0.5f);
        auto score_threshold = mm->add_literal(0.0f);

        auto r =
            mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", 1}}),
                                boxes_l,
                                scores_l,
                                max_out_l,
                                iou_threshold,
                                score_threshold);
        mm->add_return({r});

        return p;
    }
};

