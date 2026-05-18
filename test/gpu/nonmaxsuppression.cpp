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
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

#include <test.hpp>

static std::pair<std::vector<int64_t>, int64_t>
run_gpu_nms(migraphx::program p, const migraphx::parameter_map& host_params = {})
{
    migraphx::target t = migraphx::make_target("gpu");
    p.compile(t);

    migraphx::parameter_map gpu_params;
    for(auto&& x : p.get_parameter_shapes())
    {
        auto it = host_params.find(x.first);
        if(it != host_params.end())
            gpu_params[x.first] = t.copy_to(it->second);
        else
            gpu_params[x.first] = t.allocate(x.second);
    }

    auto results  = p.eval(gpu_params);
    auto idx_host = t.copy_from(results.at(0));
    auto cnt_host = t.copy_from(results.at(1));

    std::vector<int64_t> indices;
    idx_host.visit([&](auto v) { indices.assign(v.begin(), v.end()); });

    int64_t num_selected = 0;
    cnt_host.visit([&](auto v) { num_selected = static_cast<int64_t>(v[0]); });

    return {indices, num_selected};
}

static void add_nms_return(migraphx::module* mm, migraphx::instruction_ref nms)
{
    auto idx = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nms);
    auto cnt = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nms);
    mm->add_return({idx, cnt});
}

TEST_CASE(nms_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
    std::vector<float> boxes_vec = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                    0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto boxes_l         = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto nms =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            boxes_l,
                            scores_l,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    add_nms_return(mm, nms);

    auto [indices, num_selected] = run_gpu_nms(std::move(p));
    indices.resize(static_cast<std::size_t>(num_selected) * 3);
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    EXPECT(migraphx::verify::verify_rms_range(indices, gold));
    EXPECT(num_selected == 3);
}

TEST_CASE(nms_identical_all_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
    std::vector<float> boxes_vec = {0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5,
                                    0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5};
    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.9, 0.9, 0.9, 0.9, 0.9};

    auto boxes_l         = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{6});
    auto iou_threshold   = mm->add_literal(0.1f);
    auto score_threshold = mm->add_literal(0.0f);

    auto nms = mm->add_instruction(migraphx::make_op("nonmaxsuppression"),
                                   boxes_l,
                                   scores_l,
                                   max_out_l,
                                   iou_threshold,
                                   score_threshold);
    add_nms_return(mm, nms);

    auto [indices, num_selected] = run_gpu_nms(std::move(p));
    indices.resize(static_cast<std::size_t>(num_selected) * 3);
    std::vector<int64_t> gold = {0, 0, 0};
    EXPECT(migraphx::verify::verify_rms_range(indices, gold));
    EXPECT(num_selected == 1);
}

TEST_CASE(nms_not_center_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
    std::vector<float> boxes_vec = {1.0, 1.0,  0.0, 0.0,  0.0, 0.1,   1.0, 1.1,
                                    0.0, 0.9,  1.0, -0.1, 0.0, 10.0,  1.0, 11.0,
                                    1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0};

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto boxes_l         = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto nms = mm->add_instruction(migraphx::make_op("nonmaxsuppression"),
                                   boxes_l,
                                   scores_l,
                                   max_out_l,
                                   iou_threshold,
                                   score_threshold);
    add_nms_return(mm, nms);

    auto [indices, num_selected] = run_gpu_nms(std::move(p));
    indices.resize(static_cast<std::size_t>(num_selected) * 3);
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    EXPECT(migraphx::verify::verify_rms_range(indices, gold));
    EXPECT(num_selected == 3);
}

TEST_CASE(nms_transpose1_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 4, 6}};
    std::vector<float> boxes_vec = {
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.4, 10.5, 10.6, 100.5,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,
    };

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto t_boxes_l       = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto transpose_boxes = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), t_boxes_l);
    auto nms =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            transpose_boxes,
                            scores_l,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    add_nms_return(mm, nms);

    auto [indices, num_selected] = run_gpu_nms(std::move(p));
    indices.resize(static_cast<std::size_t>(num_selected) * 3);
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    EXPECT(migraphx::verify::verify_rms_range(indices, gold));
    EXPECT(num_selected == 3);
}

TEST_CASE(nms_transpose2_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {4, 1, 6}};
    std::vector<float> boxes_vec = {
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.4, 10.5, 10.6, 100.5,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,
    };

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto t_boxes_l       = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto transpose_boxes = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), t_boxes_l);
    auto nms =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            transpose_boxes,
                            scores_l,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    add_nms_return(mm, nms);

    auto [indices, num_selected] = run_gpu_nms(std::move(p));
    indices.resize(static_cast<std::size_t>(num_selected) * 3);
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5};
    EXPECT(migraphx::verify::verify_rms_range(indices, gold));
    EXPECT(num_selected == 3);
}

TEST_CASE(nms_multi_batch_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {2, 6, 4}};
    std::vector<float> boxes_vec = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                    0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0,
                                    0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                    0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};

    migraphx::shape scores_s{migraphx::shape::float_type, {2, 1, 6}};
    std::vector<float> scores_vec = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto boxes_l         = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto nms =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            boxes_l,
                            scores_l,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    add_nms_return(mm, nms);

    auto [indices, num_selected] = run_gpu_nms(std::move(p));
    indices.resize(static_cast<std::size_t>(num_selected) * 3);
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 0, 5, 1, 0, 3, 1, 0, 0, 1, 0, 5};
    EXPECT(migraphx::verify::verify_rms_range(indices, gold));
    EXPECT(num_selected == 6);
}

TEST_CASE(nms_multi_class_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};
    std::vector<float> boxes_vec = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                    0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                    0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 2, 6}};
    std::vector<float> scores_vec = {
        0.9, 0.75, 0.6, 0.95, 0.5, 0.3, 0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto boxes_l         = mm->add_literal(migraphx::literal(boxes_s, boxes_vec));
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{2});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto nms =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            boxes_l,
                            scores_l,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    add_nms_return(mm, nms);

    auto [indices, num_selected] = run_gpu_nms(std::move(p));
    indices.resize(static_cast<std::size_t>(num_selected) * 3);
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0};
    EXPECT(migraphx::verify::verify_rms_range(indices, gold));
    EXPECT(num_selected == 4);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
