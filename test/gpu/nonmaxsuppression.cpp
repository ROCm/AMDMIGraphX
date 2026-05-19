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
    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    migraphx::shape scalar_s{migraphx::shape::float_type, {1}};
    migraphx::shape int_scalar_s{migraphx::shape::int64_type, {1}};

    auto boxes_p         = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_p       = mm->add_parameter("max_out", int_scalar_s);
    auto iou_threshold   = mm->add_parameter("iou_threshold", scalar_s);
    auto score_threshold = mm->add_parameter("score_threshold", scalar_s);

    auto nms =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            boxes_p,
                            scores_p,
                            max_out_p,
                            iou_threshold,
                            score_threshold);
    add_nms_return(mm, nms);

    std::vector<float> boxes_vec = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                    0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};
    std::vector<float> scores_vec = {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f};
    int64_t max_out_val = 4;
    float iou_val       = 0.5f;
    float score_val     = 0.0f;

    migraphx::parameter_map host_params;
    host_params["boxes"]           = migraphx::argument(boxes_s, boxes_vec.data());
    host_params["scores"]          = migraphx::argument(scores_s, scores_vec.data());
    host_params["max_out"]         = migraphx::argument(int_scalar_s, &max_out_val);
    host_params["iou_threshold"]   = migraphx::argument(scalar_s, &iou_val);
    host_params["score_threshold"] = migraphx::argument(scalar_s, &score_val);

    auto [indices, num_selected] = run_gpu_nms(std::move(p), host_params);
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
    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    migraphx::shape scalar_s{migraphx::shape::float_type, {1}};

    auto boxes_p         = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_l       = mm->add_literal(int64_t{6});
    auto iou_threshold   = mm->add_parameter("iou_threshold", scalar_s);
    auto score_threshold = mm->add_literal(0.0f);

    auto nms = mm->add_instruction(migraphx::make_op("nonmaxsuppression"),
                                   boxes_p,
                                   scores_p,
                                   max_out_l,
                                   iou_threshold,
                                   score_threshold);
    add_nms_return(mm, nms);

    std::vector<float> boxes_vec = {0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5,
                                    0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.7, 0.7, 0.7, 0.7, 0.5, 0.5};
    std::vector<float> scores_vec = {0.9f, 0.9f, 0.9f, 0.9f, 0.9f, 0.9f};
    float iou_val = 0.1f;

    migraphx::parameter_map host_params;
    host_params["boxes"]         = migraphx::argument(boxes_s, boxes_vec.data());
    host_params["scores"]        = migraphx::argument(scores_s, scores_vec.data());
    host_params["iou_threshold"] = migraphx::argument(scalar_s, &iou_val);

    auto [indices, num_selected] = run_gpu_nms(std::move(p), host_params);
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
    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};

    auto boxes_p         = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto nms = mm->add_instruction(migraphx::make_op("nonmaxsuppression"),
                                   boxes_p,
                                   scores_p,
                                   max_out_l,
                                   iou_threshold,
                                   score_threshold);
    add_nms_return(mm, nms);

    std::vector<float> boxes_vec = {1.0, 1.0,  0.0, 0.0,  0.0, 0.1,   1.0, 1.1,
                                    0.0, 0.9,  1.0, -0.1, 0.0, 10.0,  1.0, 11.0,
                                    1.0, 10.1, 0.0, 11.1, 1.0, 101.0, 0.0, 100.0};
    std::vector<float> scores_vec = {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f};

    migraphx::parameter_map host_params;
    host_params["boxes"]  = migraphx::argument(boxes_s, boxes_vec.data());
    host_params["scores"] = migraphx::argument(scores_s, scores_vec.data());

    auto [indices, num_selected] = run_gpu_nms(std::move(p), host_params);
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
    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    migraphx::shape int_scalar_s{migraphx::shape::int64_type, {1}};

    auto t_boxes_p       = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_p       = mm->add_parameter("max_out", int_scalar_s);
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto transpose_boxes = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), t_boxes_p);
    auto nms =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            transpose_boxes,
                            scores_p,
                            max_out_p,
                            iou_threshold,
                            score_threshold);
    add_nms_return(mm, nms);

    std::vector<float> boxes_vec = {
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.4, 10.5, 10.6, 100.5,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,
    };
    std::vector<float> scores_vec = {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f};
    int64_t max_out_val = 4;

    migraphx::parameter_map host_params;
    host_params["boxes"]   = migraphx::argument(boxes_s, boxes_vec.data());
    host_params["scores"]  = migraphx::argument(scores_s, scores_vec.data());
    host_params["max_out"] = migraphx::argument(int_scalar_s, &max_out_val);

    auto [indices, num_selected] = run_gpu_nms(std::move(p), host_params);
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
    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};

    auto t_boxes_p       = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto transpose_boxes = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {1, 2, 0}}}), t_boxes_p);
    auto nms =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            transpose_boxes,
                            scores_p,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    add_nms_return(mm, nms);

    std::vector<float> boxes_vec = {
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6, 0.4, 10.5, 10.6, 100.5,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0,  1.0,
    };
    std::vector<float> scores_vec = {0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f};

    migraphx::parameter_map host_params;
    host_params["boxes"]  = migraphx::argument(boxes_s, boxes_vec.data());
    host_params["scores"] = migraphx::argument(scores_s, scores_vec.data());

    auto [indices, num_selected] = run_gpu_nms(std::move(p), host_params);
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
    migraphx::shape scores_s{migraphx::shape::float_type, {2, 1, 6}};
    migraphx::shape scalar_s{migraphx::shape::float_type, {1}};
    migraphx::shape int_scalar_s{migraphx::shape::int64_type, {1}};

    auto boxes_p         = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_p       = mm->add_parameter("max_out", int_scalar_s);
    auto iou_threshold   = mm->add_parameter("iou_threshold", scalar_s);
    auto score_threshold = mm->add_parameter("score_threshold", scalar_s);

    auto nms =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            boxes_p,
                            scores_p,
                            max_out_p,
                            iou_threshold,
                            score_threshold);
    add_nms_return(mm, nms);

    std::vector<float> boxes_vec = {0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                    0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0,
                                    0.5, 0.5,  1.0, 1.0, 0.5, 0.6,  1.0, 1.0, 0.5, 0.4,   1.0, 1.0,
                                    0.5, 10.5, 1.0, 1.0, 0.5, 10.6, 1.0, 1.0, 0.5, 100.5, 1.0, 1.0};
    std::vector<float> scores_vec = {
        0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f, 0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f};
    int64_t max_out_val = 4;
    float iou_val       = 0.5f;
    float score_val     = 0.0f;

    migraphx::parameter_map host_params;
    host_params["boxes"]           = migraphx::argument(boxes_s, boxes_vec.data());
    host_params["scores"]          = migraphx::argument(scores_s, scores_vec.data());
    host_params["max_out"]         = migraphx::argument(int_scalar_s, &max_out_val);
    host_params["iou_threshold"]   = migraphx::argument(scalar_s, &iou_val);
    host_params["score_threshold"] = migraphx::argument(scalar_s, &score_val);

    auto [indices, num_selected] = run_gpu_nms(std::move(p), host_params);
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
    migraphx::shape scores_s{migraphx::shape::float_type, {1, 2, 6}};
    migraphx::shape scalar_s{migraphx::shape::float_type, {1}};

    auto boxes_p         = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_l       = mm->add_literal(int64_t{2});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_parameter("score_threshold", scalar_s);

    auto nms =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            boxes_p,
                            scores_p,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    add_nms_return(mm, nms);

    std::vector<float> boxes_vec = {0.0, 0.0,  1.0, 1.0,  0.0, 0.1,   1.0, 1.1,
                                    0.0, -0.1, 1.0, 0.9,  0.0, 10.0,  1.0, 11.0,
                                    0.0, 10.1, 1.0, 11.1, 0.0, 100.0, 1.0, 101.0};
    std::vector<float> scores_vec = {
        0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f, 0.9f, 0.75f, 0.6f, 0.95f, 0.5f, 0.3f};
    float score_val = 0.0f;

    migraphx::parameter_map host_params;
    host_params["boxes"]           = migraphx::argument(boxes_s, boxes_vec.data());
    host_params["scores"]          = migraphx::argument(scores_s, scores_vec.data());
    host_params["score_threshold"] = migraphx::argument(scalar_s, &score_val);

    auto [indices, num_selected] = run_gpu_nms(std::move(p), host_params);
    indices.resize(static_cast<std::size_t>(num_selected) * 3);
    std::vector<int64_t> gold = {0, 0, 3, 0, 0, 0, 0, 1, 3, 0, 1, 0};
    EXPECT(migraphx::verify::verify_rms_range(indices, gold));
    EXPECT(num_selected == 4);
}

TEST_CASE(nms_20boxes_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 20, 4}};
    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 20}};

    auto boxes_p         = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_l       = mm->add_literal(int64_t{10});
    auto iou_threshold   = mm->add_literal(0.5000f);
    auto score_threshold = mm->add_literal(0.0000f);

    auto nms =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression"),
                            boxes_p,
                            scores_p,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    add_nms_return(mm, nms);

    std::vector<float> boxes_vec = {
        32.7256f, 35.1377f, 43.0832f, 42.2579f, 13.9286f, 15.6152f, 21.5240f, 28.2727f, 44.0782f, 37.5280f, 52.9916f, 48.3318f,
        38.8011f, 32.1818f, 50.5110f, 37.5550f, 33.9761f, -1.6170f, 43.8622f, 11.0347f, 5.3569f, 42.6478f, 14.1070f, 54.9145f,
        18.9216f, 34.8446f, 27.7505f, 41.2693f, -0.4375f, 36.7849f, 4.8178f, 41.8215f, 6.9987f, 1.1282f, 8.4302f, 11.6832f,
        30.5954f, 21.0410f, 37.7095f, 23.9976f, 35.2360f, 16.6405f, 39.2402f, 20.4393f, 45.0158f, 45.7867f, 51.7352f, 46.8898f,
        9.8174f, 26.1848f, 22.7651f, 38.2017f, 16.3854f, 35.9841f, 20.6606f, 46.2920f, 22.5697f, 16.7346f, 24.3859f, 27.6069f,
        7.0039f, 5.3968f, 11.9433f, 17.3270f, 3.9409f, 24.0168f, 9.0512f, 31.4417f, 18.6518f, -1.2903f, 28.9187f, 7.6721f,
        6.9462f, 39.9030f, 15.7447f, 42.8601f, 27.5034f, 30.2815f, 39.4780f, 32.8849f};
    std::vector<float> scores_vec = {
        0.6979f, 0.4657f, 0.8326f, 0.2503f, 0.1204f, 0.1810f, 0.7501f, 0.5157f, 0.2451f, 0.5509f, 0.2371f, 0.7267f,
        0.5015f, 0.4429f, 0.3714f, 0.6673f, 0.4256f, 0.1789f, 0.2062f, 0.9657f};

    migraphx::parameter_map host_params;
    host_params["boxes"]  = migraphx::argument(boxes_s, boxes_vec.data());
    host_params["scores"] = migraphx::argument(scores_s, scores_vec.data());

    auto [indices, num_selected] = run_gpu_nms(std::move(p), host_params);
    indices.resize(static_cast<std::size_t>(num_selected) * 3);
    std::vector<int64_t> gold = {0, 0, 19, 0, 0, 2, 0, 0, 6, 0, 0, 11, 0, 0, 0, 0, 0, 15, 0, 0, 9, 0, 0, 7, 0, 0, 12, 0, 0, 1};
    EXPECT(migraphx::verify::verify_rms_range(indices, gold));
    EXPECT(num_selected == 10);
}

TEST_CASE(nms_50boxes_center_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 50, 4}};
    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 50}};

    auto boxes_p         = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_l       = mm->add_literal(int64_t{20});
    auto iou_threshold   = mm->add_literal(0.4000f);
    auto score_threshold = mm->add_literal(0.2000f);

    auto nms =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                            boxes_p,
                            scores_p,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    add_nms_return(mm, nms);

    std::vector<float> boxes_vec = {
        90.8581f, 82.6292f, 23.5447f, 19.9060f, 69.9707f, 89.6161f, 29.1830f, 26.1572f, 26.5870f, 14.0249f, 15.5215f, 14.1630f,
        96.9176f, 55.4036f, 5.1730f, 8.1873f, 77.8751f, 10.8576f, 1.4042f, 7.8632f, 71.6890f, 67.2240f, 7.6600f, 22.6344f,
        44.9361f, 28.1234f, 4.8228f, 24.6805f, 27.2242f, 65.9423f, 20.6521f, 4.0531f, 9.6391f, 72.6995f, 4.5331f, 2.9302f,
        90.2602f, 76.8647f, 15.6836f, 18.2386f, 45.5776f, 10.7741f, 21.1336f, 5.2390f, 20.2363f, 91.6012f, 17.8524f, 24.9153f,
        30.5957f, 23.0214f, 6.7935f, 9.9997f, 57.9220f, 3.7413f, 24.3196f, 5.1723f, 17.6773f, 55.4852f, 21.7468f, 27.7081f,
        85.6614f, 37.0922f, 22.4305f, 5.8004f, 75.8520f, 82.9790f, 4.8007f, 9.2569f, 71.9463f, 80.8251f, 4.5889f, 5.4548f,
        43.2093f, 31.7139f, 27.8993f, 4.3492f, 62.7309f, 95.2899f, 12.5298f, 1.6133f, 58.4098f, 29.0918f, 9.7275f, 2.6065f,
        64.9847f, 51.5057f, 15.1689f, 6.0646f, 8.4444f, 25.5965f, 20.2231f, 2.5481f, 41.5807f, 93.6044f, 28.7131f, 18.1432f,
        4.1614f, 16.4608f, 9.3069f, 20.7407f, 49.3991f, 4.4911f, 27.8194f, 12.4153f, 32.9861f, 43.5097f, 1.7209f, 10.2217f,
        14.4524f, 99.2376f, 17.1007f, 15.6313f, 10.3403f, 89.1677f, 19.3853f, 26.3751f, 58.7645f, 74.8608f, 4.0710f, 25.6828f,
        17.0593f, 89.0792f, 5.0698f, 2.2608f, 92.5120f, 89.3447f, 13.1543f, 6.2635f, 58.1061f, 51.8858f, 29.0207f, 7.8656f,
        34.6870f, 31.5929f, 18.2852f, 8.2322f, 59.0915f, 77.2012f, 28.0577f, 17.5657f, 2.2804f, 66.1661f, 24.3265f, 13.0716f,
        95.8559f, 37.3658f, 14.5541f, 2.4284f, 48.2303f, 9.4467f, 23.7581f, 11.8348f, 78.2735f, 74.6790f, 1.5173f, 16.1888f,
        8.2730f, 26.2461f, 4.1652f, 3.9485f, 48.6658f, 93.6813f, 25.0534f, 25.1703f, 49.0707f, 24.0971f, 24.1077f, 2.5069f,
        93.7826f, 12.2758f, 7.7466f, 27.8204f, 57.1728f, 83.1113f, 16.3923f, 3.8743f, 47.3489f, 15.3284f, 18.5745f, 25.4637f,
        26.6976f, 17.9268f, 26.1644f, 27.1769f, 33.1569f, 59.9383f, 18.4901f, 29.4075f, 52.0672f, 87.4562f, 12.9646f, 24.2588f,
        43.8911f, 19.6435f, 11.8513f, 23.6048f, 2.1612f, 31.0324f, 13.3506f, 19.6320f};
    std::vector<float> scores_vec = {
        0.8011f, 0.2211f, 0.5825f, 0.5628f, 0.8718f, 0.5165f, 0.4466f, 0.6756f, 0.3398f, 0.2258f, 0.5301f, 0.4752f,
        0.3093f, 0.4308f, 0.4298f, 0.3947f, 0.4415f, 0.7172f, 0.3672f, 0.9540f, 0.9247f, 0.5328f, 0.3955f, 0.5819f,
        0.8637f, 0.6873f, 0.8240f, 0.5795f, 0.6696f, 0.3593f, 0.7614f, 0.2822f, 0.7253f, 0.8746f, 0.2189f, 0.6529f,
        0.1856f, 0.7531f, 0.1760f, 0.9423f, 0.2237f, 0.9630f, 0.8208f, 0.6343f, 0.8044f, 0.8156f, 0.9514f, 0.3280f,
        0.6311f, 0.1855f};

    migraphx::parameter_map host_params;
    host_params["boxes"]  = migraphx::argument(boxes_s, boxes_vec.data());
    host_params["scores"] = migraphx::argument(scores_s, scores_vec.data());

    auto [indices, num_selected] = run_gpu_nms(std::move(p), host_params);
    indices.resize(static_cast<std::size_t>(num_selected) * 3);
    std::vector<int64_t> gold = {0, 0, 41, 0, 0, 19, 0, 0, 46, 0, 0, 39, 0, 0, 20, 0, 0, 33, 0, 0, 4, 0, 0, 24, 0, 0, 26, 0, 0, 42, 0, 0, 45, 0, 0, 44, 0, 0, 0, 0, 0, 30, 0, 0, 32, 0, 0, 17, 0, 0, 25, 0, 0, 7, 0, 0, 28, 0, 0, 35};
    EXPECT(migraphx::verify::verify_rms_range(indices, gold));
    EXPECT(num_selected == 20);
}

TEST_CASE(nms_100boxes_2batch_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {2, 100, 4}};
    migraphx::shape scores_s{migraphx::shape::float_type, {2, 1, 100}};

    auto boxes_p         = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_l       = mm->add_literal(int64_t{15});
    auto iou_threshold   = mm->add_literal(0.5000f);
    auto score_threshold = mm->add_literal(0.1000f);

    auto nms =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression"),
                            boxes_p,
                            scores_p,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    add_nms_return(mm, nms);

    std::vector<float> boxes_vec = {
        -3.8699f, 108.8880f, 20.8101f, 137.5783f, 149.9079f, 29.3134f, 203.7504f, 39.2031f, 121.6031f, 107.1528f, 162.2282f, 118.8275f,
        27.1146f, 87.2265f, 42.1365f, 141.7457f, -7.3128f, 91.3799f, 44.0012f, 95.0142f, 25.9397f, 97.1572f, 47.4736f, 111.8955f,
        170.3318f, 143.6689f, 221.6791f, 161.9004f, 82.3933f, 144.8881f, 101.0310f, 174.8098f, 138.9017f, 80.6305f, 174.7306f, 116.2308f,
        115.0719f, 104.8666f, 139.4914f, 134.9707f, 105.8753f, 183.2658f, 123.0900f, 189.2287f, 2.3726f, 16.2585f, 55.6795f, 31.6349f,
        183.1709f, -1.9651f, 195.2389f, 48.8066f, 57.2666f, -1.7671f, 63.2705f, 36.8507f, 105.0166f, 111.9228f, 126.1903f, 151.2225f,
        118.2848f, 63.4507f, 161.6255f, 103.9927f, 105.5274f, 131.8586f, 154.1659f, 177.8699f, 158.1560f, 132.0321f, 218.0818f, 136.4605f,
        20.4451f, 55.4126f, 38.9305f, 78.0425f, 89.1363f, 163.2572f, 114.2048f, 196.0894f, 76.2707f, 142.0220f, 85.3431f, 162.9909f,
        77.3750f, 28.6949f, 112.2925f, 79.5191f, -6.0851f, 58.1025f, 53.7721f, 87.5743f, 5.6429f, 39.7135f, 47.9949f, 86.0625f,
        37.5563f, 5.8879f, 73.6739f, 57.1568f, 48.8660f, 14.1653f, 73.0158f, 44.9480f, 58.0793f, 159.8937f, 113.0820f, 214.5573f,
        107.0385f, 69.7607f, 137.3566f, 105.4010f, 122.4620f, 51.0809f, 131.3896f, 102.2471f, 71.0835f, 135.3897f, 93.6408f, 156.4846f,
        79.2752f, 95.3835f, 84.2380f, 125.8137f, 37.0673f, 171.0514f, 49.9841f, 203.4046f, 116.6400f, 152.4634f, 118.6825f, 159.6572f,
        49.5364f, 83.6166f, 77.2799f, 108.1312f, -12.0070f, 47.7104f, 26.4309f, 102.8334f, 73.0529f, 178.2168f, 94.3071f, 216.4359f,
        81.9253f, 137.8156f, 107.7278f, 149.2885f, 16.3219f, 179.7427f, 73.9152f, 200.7352f, 91.8087f, 17.5434f, 137.1745f, 29.8480f,
        96.6991f, 168.8745f, 129.6096f, 171.3390f, 131.5065f, 99.5547f, 149.2944f, 155.2749f, 102.6283f, 10.6622f, 156.5511f, 38.1065f,
        123.0512f, 108.0793f, 137.9220f, 127.2239f, 53.1452f, 119.0642f, 73.3404f, 155.3743f, 130.1690f, 1.7448f, 184.8039f, 3.1763f,
        93.7074f, 82.1619f, 125.9504f, 99.5652f, 63.8853f, 143.8404f, 108.6820f, 186.3194f, 107.2755f, 39.8756f, 143.1295f, 78.2680f,
        52.3550f, 62.2463f, 91.9079f, 121.1729f, 93.2160f, 69.6623f, 111.8797f, 107.2634f, 139.7207f, 45.7991f, 154.9616f, 74.9719f,
        167.2671f, 160.7261f, 187.2941f, 206.6506f, 179.1259f, 129.1106f, 189.2970f, 183.4070f, 74.4343f, 0.3572f, 127.0189f, 43.8782f,
        95.1992f, 170.4922f, 112.9108f, 228.3217f, 142.9101f, 152.2709f, 177.0380f, 199.4092f, 39.0269f, 30.7110f, 86.7534f, 82.8523f,
        143.8537f, 163.5132f, 191.0993f, 171.2454f, 85.3959f, -0.8223f, 112.2607f, 43.3901f, 8.6218f, 186.3383f, 37.7209f, 213.3036f,
        -15.4319f, 116.3204f, 44.2555f, 149.9535f, 147.9980f, 110.2290f, 188.7993f, 149.8210f, -13.4183f, -11.0214f, 35.6454f, 47.1977f,
        28.9969f, 149.8616f, 83.2476f, 208.9517f, 43.0921f, -3.2028f, 90.5599f, 14.8026f, 28.6361f, 26.0199f, 40.5617f, 70.3113f,
        45.6946f, 5.9799f, 79.8627f, 51.2289f, 145.0326f, 144.6320f, 152.0444f, 166.0751f, -16.8246f, 35.4867f, 22.6978f, 43.7950f,
        136.7519f, 180.4197f, 194.1175f, 183.8356f, 155.6840f, 107.8222f, 186.9352f, 154.6854f, 61.1796f, -7.7136f, 87.7250f, 22.1787f,
        29.1652f, -28.4875f, 32.2799f, 30.6594f, 91.3547f, -3.8851f, 148.9814f, 24.5483f, 20.3959f, 91.8365f, 27.4731f, 150.5336f,
        71.2720f, 147.6549f, 74.6957f, 172.9379f, 183.9269f, 23.7969f, 199.4448f, 71.6242f, 196.6597f, 166.8796f, 201.5260f, 172.8839f,
        140.4950f, -5.4397f, 168.3470f, 28.3325f, 46.4677f, 136.0320f, 77.9169f, 184.3535f, 127.8122f, 157.7804f, 147.2538f, 213.3378f,
        139.0779f, 129.6555f, 143.0846f, 179.1879f, 73.7761f, 138.0335f, 81.3605f, 141.2148f, 116.3348f, 156.1013f, 140.0206f, 179.0908f,
        -0.1401f, 6.0937f, 4.4311f, 9.9669f, 20.7149f, 36.6326f, 62.9081f, 44.0802f, 98.4106f, 4.5632f, 111.6248f, 45.4062f,
        23.3391f, 79.3651f, 42.1614f, 122.4473f, 21.0547f, 125.7129f, 45.3081f, 172.3624f, 154.4709f, 99.9714f, 180.0508f, 152.0333f,
        197.2776f, 147.9130f, 198.3756f, 192.5394f, 107.3878f, 6.9169f, 115.0000f, 55.1683f, 141.8624f, 144.9798f, 193.7655f, 148.8687f,
        197.5280f, 31.1895f, 198.6007f, 46.0271f, 12.8282f, 35.3058f, 43.8101f, 72.9977f, 74.7088f, 116.1662f, 104.5894f, 167.7956f,
        68.1883f, 195.4082f, 88.8408f, 196.6737f, 2.7857f, 106.6272f, 29.2340f, 137.9903f, 127.5389f, -9.5799f, 174.5932f, 31.3800f,
        61.4403f, 121.8884f, 112.0713f, 124.6352f, 15.4868f, 35.9096f, 55.8899f, 68.2298f, 35.5922f, 56.6701f, 44.2246f, 72.3261f,
        163.1796f, 40.7751f, 180.4136f, 56.2181f, 177.9262f, 90.7157f, 187.1069f, 101.2297f, 33.5656f, 108.4211f, 51.2933f, 164.8822f,
        73.5555f, 18.9549f, 114.3649f, 72.3462f, 119.3443f, 42.7151f, 174.0536f, 89.5792f, 169.1987f, 170.3059f, 182.1476f, 201.8479f,
        59.3192f, -5.2591f, 92.3019f, 24.6868f, 82.2129f, 76.0264f, 124.5949f, 108.2814f, 119.7321f, 125.9828f, 176.9545f, 158.6404f,
        127.7304f, 16.7712f, 164.7240f, 43.4104f, 148.5664f, 5.0880f, 164.6177f, 13.8616f, 95.0352f, 23.4340f, 132.9384f, 31.8482f,
        10.9685f, 155.1733f, 30.8775f, 212.3560f, 151.4989f, -12.8680f, 210.0904f, 16.5719f, 160.8241f, 9.0448f, 185.4050f, 66.2840f,
        138.8994f, 0.9312f, 180.3396f, 11.5822f, 18.7873f, 5.2706f, 21.1577f, 38.9812f, 28.5777f, 117.4022f, 53.1813f, 130.6575f,
        122.4044f, 40.3588f, 175.0358f, 56.2967f, -13.8737f, 112.4558f, 23.1297f, 115.2290f, 182.2486f, 114.0300f, 209.4412f, 122.0482f,
        47.3188f, 142.3400f, 103.5391f, 197.4341f, 118.1700f, -9.0369f, 169.5550f, 10.9335f, 167.5089f, 152.2341f, 187.5196f, 189.1137f,
        62.3618f, 109.6059f, 95.4902f, 138.0417f, 48.8767f, 20.2354f, 78.7763f, 44.8620f, 102.5983f, 138.3968f, 140.8982f, 170.7781f,
        105.8416f, 165.0748f, 126.5542f, 177.1219f, 74.1239f, 21.1889f, 89.5320f, 80.5165f, 92.9311f, 159.1187f, 147.7788f, 208.3988f,
        159.3220f, 68.5139f, 214.8306f, 113.2691f, 68.1500f, 106.3565f, 118.9061f, 135.0133f, 9.9914f, 191.9200f, 68.7055f, 201.9398f,
        52.9639f, 44.6476f, 97.9184f, 99.9669f, 55.7637f, 152.0609f, 101.8791f, 173.2028f, 3.2253f, 61.7017f, 49.2181f, 65.6580f,
        17.8964f, 149.2418f, 47.2522f, 170.4436f, 122.9471f, 96.2103f, 150.8778f, 144.0833f, 60.3089f, 24.4012f, 75.4822f, 62.1410f,
        171.4575f, 60.1555f, 210.5018f, 105.4550f, 39.6844f, 39.6149f, 57.7543f, 87.4394f, 11.6796f, 8.8690f, 27.8902f, 22.3743f,
        132.9151f, -21.7847f, 168.4868f, 33.7186f, 163.6127f, 55.8750f, 188.8017f, 82.7164f, 48.6664f, -15.5441f, 62.5789f, 23.1577f,
        15.8440f, 32.5294f, 64.9913f, 33.6657f, 11.2664f, 115.2323f, 63.0400f, 174.8410f, 98.9553f, 132.8318f, 109.8496f, 150.4047f,
        92.9619f, 145.3852f, 94.4048f, 150.0469f, 41.4721f, 49.4119f, 62.3038f, 77.4494f, -14.9919f, 173.6975f, 33.0612f, 182.3103f,
        71.0426f, 113.7725f, 121.5539f, 123.7598f, 187.2858f, 6.0529f, 196.4472f, 44.3576f, 107.1609f, 16.6524f, 153.8468f, 40.8351f,
        95.1880f, 110.9244f, 103.0146f, 166.3137f, 10.1316f, 24.6737f, 34.1453f, 44.5039f, 20.5283f, 79.5362f, 80.4462f, 123.3809f,
        52.7734f, 184.2525f, 65.1362f, 212.4573f, 147.9188f, -19.1670f, 158.0026f, 20.7701f, 162.3696f, -14.8751f, 188.3148f, 21.5070f,
        161.5482f, 184.1698f, 199.1086f, 213.0640f, 168.8931f, 88.4010f, 224.9343f, 145.4546f, 167.0391f, 14.7719f, 225.9076f, 35.9920f,
        188.0454f, 173.7320f, 193.1542f, 185.1889f, 9.7935f, 155.5723f, 18.9354f, 196.5798f, 3.7319f, 81.7829f, 51.3855f, 132.6973f,
        52.4097f, 122.6709f, 69.3770f, 126.0459f, 83.9766f, 40.8733f, 137.1827f, 68.4016f, -0.6763f, -16.7244f, 39.4674f, 36.9323f,
        165.3600f, 96.2998f, 172.9588f, 141.5273f, 98.2916f, 29.1927f, 148.4108f, 88.7094f, 102.7704f, 116.5475f, 114.1754f, 148.9009f,
        20.0692f, 147.2792f, 46.0554f, 187.2189f, 33.8616f, -5.7911f, 67.4406f, 13.0553f, 16.7898f, 90.6905f, 47.3350f, 147.5951f,
        149.6448f, 34.9492f, 191.1284f, 57.5630f, 97.0913f, 152.4916f, 136.5998f, 197.0638f, 117.2606f, 38.3403f, 176.7911f, 63.1255f,
        29.2236f, 105.0804f, 89.1895f, 139.2277f, 58.5150f, 88.9746f, 89.9861f, 132.4418f, 77.6626f, 63.7197f, 84.2794f, 94.7469f,
        130.0316f, 108.2651f, 173.9744f, 162.7832f, 125.1590f, 132.2845f, 183.7822f, 158.0233f, 31.4721f, 93.7989f, 51.2533f, 132.9762f,
        174.2021f, 141.0848f, 202.4134f, 162.2841f, 11.1001f, 184.1428f, 37.1620f, 209.2240f, 177.2076f, 70.3730f, 181.2413f, 97.3360f,
        -0.2527f, 98.7053f, 40.4109f, 107.1279f, 41.9845f, -0.7119f, 63.8314f, 5.6998f, 145.5655f, 139.0148f, 193.0259f, 179.3967f,
        10.8509f, 84.2082f, 60.9460f, 123.8838f, 57.9873f, 61.5364f, 107.4399f, 101.6481f, 77.1802f, 17.7313f, 102.7635f, 19.8975f,
        39.0662f, 167.7982f, 59.0374f, 188.0644f, 119.4588f, 72.6661f, 164.6393f, 85.3368f, 146.1259f, 113.0609f, 194.4079f, 159.9718f,
        159.9229f, 3.9862f, 189.9071f, 55.7634f, 41.0200f, 184.5329f, 94.7088f, 200.0870f};
    std::vector<float> scores_vec = {
        0.1439f, 0.8791f, 0.0961f, 0.1535f, 0.5338f, 0.0675f, 0.0528f, 0.0005f, 0.4363f, 0.7746f, 0.0348f, 0.6523f,
        0.8231f, 0.1680f, 0.1469f, 0.8608f, 0.8231f, 0.5389f, 0.8192f, 0.0928f, 0.3945f, 0.7378f, 0.2575f, 0.7523f,
        0.5042f, 0.7503f, 0.4647f, 0.3679f, 0.2192f, 0.2084f, 0.7515f, 0.1189f, 0.0860f, 0.1763f, 0.1753f, 0.8231f,
        0.3985f, 0.9904f, 0.1372f, 0.6535f, 0.4487f, 0.3929f, 0.8751f, 0.9756f, 0.8729f, 0.1923f, 0.2208f, 0.6561f,
        0.2891f, 0.7347f, 0.5664f, 0.5509f, 0.8285f, 0.7105f, 0.0266f, 0.0495f, 0.6016f, 0.4862f, 0.2602f, 0.4187f,
        0.7579f, 0.8266f, 0.5612f, 0.3854f, 0.2707f, 0.5219f, 0.3147f, 0.5641f, 0.6767f, 0.0661f, 0.0011f, 0.2123f,
        0.8945f, 0.6463f, 0.1720f, 0.8903f, 0.4700f, 0.4761f, 0.9355f, 0.0595f, 0.2152f, 0.5858f, 0.1955f, 0.6795f,
        0.2141f, 0.0992f, 0.2070f, 0.4227f, 0.1761f, 0.1347f, 0.8603f, 0.3204f, 0.3608f, 0.0553f, 0.3574f, 0.2648f,
        0.6105f, 0.2054f, 0.8884f, 0.9297f, 0.0998f, 0.1074f, 0.1153f, 0.6196f, 0.1220f, 0.8524f, 0.7543f, 0.8198f,
        0.5261f, 0.9967f, 0.0442f, 0.4013f, 0.3239f, 0.9486f, 0.5769f, 0.8062f, 0.1703f, 0.9786f, 0.4986f, 0.4937f,
        0.9709f, 0.3807f, 0.3975f, 0.5848f, 0.1281f, 0.3211f, 0.1932f, 0.1033f, 0.8661f, 0.5893f, 0.3587f, 0.4087f,
        0.4315f, 0.6331f, 0.9268f, 0.9328f, 0.3915f, 0.3293f, 0.4510f, 0.5679f, 0.4618f, 0.6588f, 0.5544f, 0.3207f,
        0.3457f, 0.3786f, 0.0946f, 0.1661f, 0.7231f, 0.3891f, 0.2145f, 0.5627f, 0.7555f, 0.2574f, 0.8268f, 0.9275f,
        0.5974f, 0.6689f, 0.0526f, 0.9455f, 0.3925f, 0.9239f, 0.5790f, 0.0046f, 0.0385f, 0.6804f, 0.5627f, 0.0265f,
        0.7435f, 0.8521f, 0.4964f, 0.4658f, 0.0055f, 0.7866f, 0.3307f, 0.8788f, 0.3731f, 0.5651f, 0.2703f, 0.1606f,
        0.7749f, 0.4966f, 0.5365f, 0.9654f, 0.9636f, 0.8556f, 0.1876f, 0.5943f, 0.8781f, 0.3745f, 0.1011f, 0.8110f,
        0.4818f, 0.5644f, 0.9821f, 0.6072f, 0.4250f, 0.3700f, 0.4176f, 0.1184f};

    migraphx::parameter_map host_params;
    host_params["boxes"]  = migraphx::argument(boxes_s, boxes_vec.data());
    host_params["scores"] = migraphx::argument(scores_s, scores_vec.data());

    auto [indices, num_selected] = run_gpu_nms(std::move(p), host_params);
    indices.resize(static_cast<std::size_t>(num_selected) * 3);
    std::vector<int64_t> gold = {0, 0, 37, 0, 0, 43, 0, 0, 78, 0, 0, 99, 0, 0, 72, 0, 0, 75, 0, 0, 98, 0, 0, 1, 0, 0, 42, 0, 0, 44, 0, 0, 15, 0, 0, 90, 0, 0, 52, 0, 0, 61, 0, 0, 12, 1, 0, 9, 1, 0, 94, 1, 0, 17, 1, 0, 20, 1, 0, 83, 1, 0, 84, 1, 0, 13, 1, 0, 59, 1, 0, 35, 1, 0, 55, 1, 0, 34, 1, 0, 61, 1, 0, 75, 1, 0, 88, 1, 0, 28};
    EXPECT(migraphx::verify::verify_rms_range(indices, gold));
    EXPECT(num_selected == 30);
}

TEST_CASE(nms_30boxes_3class_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 30, 4}};
    migraphx::shape scores_s{migraphx::shape::float_type, {1, 3, 30}};

    auto boxes_p         = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_l       = mm->add_literal(int64_t{5});
    auto iou_threshold   = mm->add_literal(0.4500f);
    auto score_threshold = mm->add_literal(0.1500f);

    auto nms =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression"),
                            boxes_p,
                            scores_p,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    add_nms_return(mm, nms);

    std::vector<float> boxes_vec = {
        31.2680f, 53.5348f, 37.7043f, 73.6253f, 1.8071f, 55.2945f, 3.9368f, 78.7402f, 40.5016f, 12.5670f, 45.0345f, 32.9366f,
        78.2552f, 12.9548f, 80.7117f, 35.6526f, 73.9527f, 67.9870f, 79.4405f, 71.9065f, -3.8066f, -7.7339f, 10.2705f, 11.5692f,
        45.4706f, 34.8613f, 67.4569f, 48.4119f, 17.4632f, 30.3439f, 30.8192f, 43.8443f, 64.5403f, 44.3725f, 79.9380f, 66.0477f,
        0.7877f, 1.3956f, 6.4307f, 24.7471f, 65.1632f, 44.8608f, 84.5766f, 62.0721f, 59.3935f, 24.0849f, 74.6026f, 36.1925f,
        -1.0372f, 43.7485f, 19.8379f, 55.2458f, -6.6257f, -1.7353f, 16.1976f, 8.1505f, 62.2758f, 32.2798f, 71.2775f, 41.5966f,
        10.9190f, 36.7777f, 14.0023f, 46.7824f, 39.6937f, 15.6139f, 45.8900f, 18.6783f, 67.7244f, 9.7794f, 78.7948f, 12.5604f,
        34.0204f, 5.6094f, 56.7713f, 24.5464f, 26.9281f, 21.9014f, 36.6292f, 33.1611f, 26.2374f, -3.4581f, 44.9652f, 18.9477f,
        -1.6661f, 68.2450f, 11.7649f, 83.3261f, 74.8979f, 31.4950f, 80.1025f, 33.3041f, 20.6639f, 62.4061f, 29.0408f, 67.0291f,
        7.1374f, 75.0864f, 23.1608f, 80.8203f, 14.6460f, -5.2621f, 31.1216f, 18.1798f, 71.6501f, 49.1185f, 82.6496f, 55.1487f,
        4.4135f, 63.2815f, 10.6723f, 76.1439f, 60.5823f, 39.4727f, 78.1862f, 62.0048f, 54.1855f, 22.5844f, 59.0696f, 46.0598f};
    std::vector<float> scores_vec = {
        0.9367f, 0.1879f, 0.1073f, 0.4976f, 0.5195f, 0.5082f, 0.4367f, 0.9948f, 0.4863f, 0.4779f, 0.4218f, 0.0668f,
        0.5930f, 0.2280f, 0.6376f, 0.0508f, 0.9814f, 0.4690f, 0.8968f, 0.4756f, 0.0603f, 0.8222f, 0.6482f, 0.7818f,
        0.4282f, 0.6379f, 0.8562f, 0.6311f, 0.3477f, 0.6625f, 0.6719f, 0.9606f, 0.3709f, 0.4251f, 0.8121f, 0.5058f,
        0.7366f, 0.4597f, 0.2155f, 0.7452f, 0.1312f, 0.1986f, 0.6268f, 0.7473f, 0.8947f, 0.2726f, 0.1107f, 0.9560f,
        0.1544f, 0.1977f, 0.2913f, 0.5294f, 0.8828f, 0.7605f, 0.7082f, 0.1752f, 0.3577f, 0.4784f, 0.1474f, 0.2734f,
        0.3083f, 0.1273f, 0.5502f, 0.7050f, 0.0699f, 0.4811f, 0.7822f, 0.7480f, 0.8151f, 0.4482f, 0.8206f, 0.2408f,
        0.3608f, 0.1764f, 0.4675f, 0.3921f, 0.2409f, 0.7518f, 0.3138f, 0.2728f, 0.1309f, 0.4388f, 0.3030f, 0.3693f,
        0.2360f, 0.7632f, 0.9300f, 0.4979f, 0.6430f, 0.8672f};

    migraphx::parameter_map host_params;
    host_params["boxes"]  = migraphx::argument(boxes_s, boxes_vec.data());
    host_params["scores"] = migraphx::argument(scores_s, scores_vec.data());

    auto [indices, num_selected] = run_gpu_nms(std::move(p), host_params);
    indices.resize(static_cast<std::size_t>(num_selected) * 3);
    std::vector<int64_t> gold = {0, 0, 7, 0, 0, 16, 0, 0, 0, 0, 0, 18, 0, 0, 26, 0, 1, 1, 0, 1, 17, 0, 1, 14, 0, 1, 22, 0, 1, 4, 0, 2, 26, 0, 2, 29, 0, 2, 10, 0, 2, 6, 0, 2, 25};
    EXPECT(migraphx::verify::verify_rms_range(indices, gold));
    EXPECT(num_selected == 15);
}

TEST_CASE(nms_200boxes_2batch_2class_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape boxes_s{migraphx::shape::float_type, {2, 200, 4}};
    migraphx::shape scores_s{migraphx::shape::float_type, {2, 2, 200}};

    auto boxes_p         = mm->add_parameter("boxes", boxes_s);
    auto scores_p        = mm->add_parameter("scores", scores_s);
    auto max_out_l       = mm->add_literal(int64_t{25});
    auto iou_threshold   = mm->add_literal(0.3000f);
    auto score_threshold = mm->add_literal(0.2500f);

    auto nms =
        mm->add_instruction(migraphx::make_op("nonmaxsuppression"),
                            boxes_p,
                            scores_p,
                            max_out_l,
                            iou_threshold,
                            score_threshold);
    add_nms_return(mm, nms);

    std::vector<float> boxes_vec = {
        132.1894f, 453.1169f, 199.9736f, 545.7127f, 64.3090f, 275.1729f, 104.8258f, 338.3436f, 76.1273f, 401.7875f, 135.6448f, 487.9920f,
        12.8305f, 442.3624f, 77.1708f, 466.2458f, -5.9609f, 340.1129f, 126.0715f, 451.3386f, 15.0119f, 224.3769f, 56.2927f, 236.5545f,
        427.8277f, -14.2917f, 561.9954f, 95.4457f, 4.7940f, -55.8461f, 69.2637f, 71.6517f, 41.3494f, 202.9014f, 91.1927f, 274.2992f,
        375.6902f, 208.6749f, 451.5645f, 285.6396f, 258.4982f, 179.9212f, 321.7420f, 227.4412f, 367.5344f, 211.3590f, 406.8828f, 356.8083f,
        277.5064f, 220.9636f, 353.4056f, 331.1991f, 429.2783f, 390.3169f, 452.8968f, 446.2962f, 292.5150f, 40.8054f, 345.9525f, 67.8517f,
        218.4112f, 95.7302f, 303.7139f, 129.4475f, 325.0759f, 361.4403f, 387.6738f, 431.5647f, 161.8149f, 353.1971f, 285.5779f, 494.6398f,
        153.4061f, 442.2182f, 192.6577f, 552.6060f, 161.0782f, 419.9203f, 306.5742f, 452.9917f, 25.8953f, 380.4122f, 143.8188f, 509.4868f,
        325.7002f, 128.4980f, 470.8716f, 185.8499f, 67.4107f, 136.8775f, 193.2931f, 264.7841f, 65.6790f, 115.5359f, 87.8525f, 152.5492f,
        83.4548f, 256.5595f, 162.8974f, 349.7399f, 407.8717f, 399.8657f, 434.1985f, 538.9396f, 103.6427f, 152.6073f, 226.5586f, 192.0336f,
        299.0049f, 226.3779f, 387.0450f, 330.6239f, 408.0779f, 74.0950f, 448.3318f, 222.2046f, -30.8828f, 73.1804f, 108.6275f, 96.6196f,
        373.4308f, 90.5068f, 391.5936f, 104.6787f, 111.3250f, -21.7549f, 196.3405f, 79.7002f, 54.0937f, 448.8364f, 162.5287f, 500.4571f,
        339.5665f, 195.6321f, 349.3349f, 207.2475f, 409.8580f, 381.1502f, 499.9386f, 452.9707f, 86.2250f, 284.0088f, 208.7943f, 397.3206f,
        278.8861f, 74.2190f, 289.9477f, 117.7022f, 106.2550f, 62.2701f, 183.5792f, 113.1921f, 257.3803f, 342.4895f, 296.9053f, 469.4987f,
        261.0432f, 93.1105f, 360.8189f, 171.6012f, 295.8262f, 393.3591f, 314.5092f, 519.9261f, 241.4629f, 36.2717f, 382.0835f, 103.7837f,
        0.3826f, 267.3577f, 134.6972f, 410.3510f, 332.4151f, 358.2527f, 361.1253f, 456.2211f, 312.7919f, 108.4937f, 361.9585f, 126.7627f,
        297.0153f, 71.6643f, 385.5729f, 204.5431f, -16.9604f, 445.3092f, 91.0309f, 519.2097f, 189.9415f, 121.2467f, 256.8973f, 143.3509f,
        192.3739f, 203.1031f, 216.6613f, 226.8539f, 35.0965f, 164.5365f, 51.6150f, 267.9791f, 36.2014f, 122.4881f, 186.1665f, 130.5466f,
        186.0576f, 366.0443f, 254.9050f, 409.7468f, 305.9496f, 375.0105f, 436.9568f, 396.8388f, 82.0940f, 155.7987f, 154.9680f, 222.5193f,
        345.6593f, 386.1935f, 484.0906f, 448.9323f, 265.8611f, 67.1577f, 279.9372f, 145.9173f, 371.2164f, -19.1800f, 389.2053f, 23.4858f,
        166.5204f, 282.6964f, 306.0356f, 288.4709f, 178.5089f, 450.7671f, 320.6853f, 543.3107f, 285.9132f, -9.0198f, 333.8062f, 47.6641f,
        437.0255f, 54.9746f, 490.9451f, 153.0235f, 211.6987f, 250.8616f, 280.1138f, 268.0530f, 232.8247f, 403.4440f, 295.8328f, 406.4968f,
        286.3401f, 25.5231f, 315.6569f, 63.5189f, 301.3286f, 163.1046f, 436.1865f, 232.1301f, 16.5538f, 343.6795f, 55.2966f, 403.3963f,
        204.8009f, 124.9041f, 310.8865f, 246.6391f, 235.2927f, 65.7693f, 246.2989f, 123.0671f, 457.4555f, 57.7300f, 464.2295f, 137.7658f,
        197.5504f, 160.3075f, 295.9562f, 249.7413f, 208.4036f, 237.5821f, 259.9170f, 241.8350f, 431.7683f, 392.0298f, 530.4317f, 469.7846f,
        217.7836f, 294.9363f, 232.7928f, 347.3161f, 19.1783f, 313.3156f, 161.7061f, 377.0863f, 52.1937f, 483.5222f, 164.7224f, 499.4650f,
        -18.1881f, 147.1016f, 113.3757f, 264.7419f, -10.3830f, 130.9681f, 10.9511f, 272.3863f, 191.6208f, 459.5145f, 240.3248f, 463.8325f,
        356.6797f, 77.6355f, 412.5629f, 168.2401f, 326.2139f, 307.5013f, 407.2526f, 422.3140f, -6.5422f, 355.5684f, 38.6912f, 399.0047f,
        279.9745f, -10.2789f, 290.0085f, 108.0669f, 49.1601f, 186.5052f, 105.1230f, 281.7262f, 451.0742f, 30.5586f, 490.0021f, 170.0038f,
        54.4314f, 19.1028f, 112.9336f, 166.2725f, 298.1461f, 228.2593f, 328.4931f, 235.5688f, 143.1079f, 111.0670f, 183.1305f, 178.3627f,
        273.5727f, 356.7796f, 367.9886f, 439.2808f, 176.7118f, 442.3701f, 235.5468f, 465.2348f, 353.5905f, 375.8070f, 406.0526f, 426.9136f,
        75.0636f, 58.9357f, 155.6155f, 207.0952f, 394.8923f, 135.3580f, 510.8995f, 138.7764f, 221.3792f, 93.1523f, 278.8305f, 161.5760f,
        333.7764f, 4.2413f, 422.3168f, 130.7968f, 352.3830f, 447.2686f, 497.3472f, 496.5298f, 460.0268f, 164.7789f, 538.8018f, 237.2689f,
        43.6929f, 38.9803f, 180.2527f, 185.7092f, 83.8176f, 387.4572f, 203.0748f, 459.2138f, 120.3420f, 189.3440f, 130.0911f, 209.8513f,
        98.9678f, 13.2052f, 163.9035f, 21.9117f, 238.6976f, 10.0373f, 343.7471f, 151.9043f, 422.7512f, 299.3224f, 570.7713f, 339.9280f,
        460.4900f, 353.3999f, 529.7881f, 429.5054f, 255.9741f, 98.2099f, 270.7991f, 112.7245f, 277.1439f, 426.6355f, 361.8833f, 490.7601f,
        420.0563f, 355.7057f, 439.9143f, 495.2914f, 409.9785f, 386.2606f, 522.9550f, 462.1201f, 63.6084f, 40.9810f, 140.2522f, 186.6801f,
        209.8752f, 5.4847f, 318.6665f, 45.0513f, 351.1511f, 395.6231f, 481.6860f, 471.8004f, 104.2444f, 88.3651f, 198.9577f, 217.4352f,
        173.7778f, 275.5634f, 266.0312f, 343.3530f, 436.0951f, 358.6616f, 549.5261f, 401.3052f, 429.2604f, -0.0863f, 555.7863f, 128.3795f,
        387.8089f, 360.8724f, 518.2979f, 419.9659f, 396.0101f, 429.2169f, 402.4382f, 509.2946f, 92.6291f, 290.9362f, 176.5014f, 437.4388f,
        143.8130f, 206.2184f, 177.0371f, 235.0044f, 209.0457f, 415.3847f, 338.2372f, 461.2934f, 231.5831f, 260.9141f, 329.1943f, 266.5435f,
        220.9448f, 342.6935f, 284.5580f, 402.0774f, 303.8214f, 394.8393f, 332.8489f, 425.6666f, 178.4043f, 323.5138f, 229.9188f, 425.8390f,
        321.6556f, 129.9190f, 427.5185f, 157.9359f, 151.0502f, 8.1484f, 182.4998f, 109.6955f, 157.8666f, 99.0403f, 172.8104f, 139.2982f,
        -3.0452f, 224.4737f, 130.2711f, 278.4012f, 36.9224f, 226.1483f, 151.7898f, 279.1286f, 409.8757f, 237.4242f, 440.6452f, 345.2202f,
        200.8640f, 162.2960f, 245.4184f, 232.8059f, 41.0147f, 366.0289f, 186.8531f, 420.8625f, 326.4108f, 392.5565f, 432.9303f, 520.5973f,
        231.0067f, 80.2522f, 322.9745f, 166.4729f, -12.8403f, 351.8312f, 33.9963f, 384.6920f, 135.3959f, 271.4291f, 180.9655f, 406.5427f,
        85.0562f, 235.5178f, 91.9452f, 287.5727f, 273.1645f, 90.8612f, 382.7083f, 97.6691f, 133.7990f, 360.2684f, 141.2321f, 434.9638f,
        31.6115f, 470.5798f, 33.3353f, 490.0465f, -27.3799f, 342.6524f, 82.3149f, 379.1839f, 219.6726f, 402.7702f, 362.0547f, 515.0898f,
        -45.9977f, 481.8516f, 67.7212f, 502.3336f, 388.7589f, 115.4080f, 460.0333f, 236.6427f, 40.9882f, 248.8122f, 114.4089f, 389.4114f,
        270.2910f, 191.2797f, 336.2753f, 282.6530f, 197.6581f, 439.8926f, 247.0300f, 546.7361f, 182.0580f, -6.7583f, 260.7935f, 100.5661f,
        3.2778f, 131.7233f, 68.5193f, 280.6516f, 356.3126f, 411.8249f, 446.4396f, 463.7141f, 379.1163f, 129.3928f, 513.9362f, 154.6585f,
        -69.1199f, 354.7185f, 80.1365f, 433.0744f, 82.9357f, 151.1645f, 95.6685f, 231.6187f, 422.7932f, 476.2348f, 481.1110f, 503.7437f,
        260.7842f, 395.5883f, 288.7094f, 487.9416f, 48.2868f, 149.1079f, 101.7528f, 152.2125f, 79.4785f, 315.4853f, 123.3120f, 454.7079f,
        316.4901f, 148.2175f, 343.4961f, 188.6391f, 304.9847f, 299.7342f, 419.8321f, 306.6287f, 262.2399f, 320.6758f, 337.1869f, 337.8050f,
        407.5904f, 396.3992f, 545.5580f, 433.1963f, 244.1037f, -8.6806f, 249.9599f, 33.1314f, 144.6461f, 107.1346f, 155.6258f, 113.0233f,
        208.0726f, 334.6470f, 269.1603f, 377.2708f, 173.3525f, 266.8875f, 186.3138f, 296.6358f, 92.1346f, 219.0953f, 132.2813f, 276.5098f,
        -50.9776f, -1.5900f, 96.9408f, 56.8000f, 160.0388f, 148.3819f, 192.1737f, 199.8940f, 340.4449f, 407.6198f, 370.9644f, 457.4804f,
        -34.0173f, 8.2614f, 52.4551f, 22.6314f, 181.9884f, 195.8403f, 257.1901f, 200.5959f, 278.2621f, 457.0166f, 365.7473f, 488.1317f,
        276.6353f, -31.4300f, 333.7688f, 82.3108f, 326.2304f, 300.5375f, 450.4180f, 449.1682f, 394.4356f, 59.1311f, 416.0841f, 198.4815f,
        323.4377f, 395.2401f, 388.2682f, 471.3687f, -0.4884f, 332.9131f, 103.2861f, 413.1549f, 172.3276f, 418.9163f, 302.6948f, 466.7889f,
        273.6699f, 49.8039f, 329.7361f, 166.1209f, 79.9860f, 208.1720f, 165.5801f, 323.1208f, 15.6250f, 326.2367f, 26.9268f, 453.0333f,
        98.6064f, 55.6348f, 124.9839f, 190.0650f, 221.7964f, 82.5141f, 233.0980f, 148.2322f, 152.2380f, -44.0412f, 261.6923f, 71.2233f,
        66.3730f, 418.6809f, 110.2940f, 539.8344f, 357.7888f, 331.5282f, 466.6268f, 378.4887f, 457.3967f, 248.0516f, 468.2900f, 387.5087f,
        35.9143f, 364.4689f, 165.4340f, 379.5258f, 402.0395f, 191.2334f, 527.5334f, 340.3795f, 1.8053f, 180.1951f, 16.0557f, 295.9387f,
        460.2114f, 217.3174f, 464.7511f, 232.2148f, 471.2709f, 270.8305f, 480.6579f, 369.6087f, -58.0695f, 97.7211f, 70.1214f, 103.8139f,
        363.5242f, 386.1504f, 399.4951f, 501.9083f, 443.7544f, 345.8341f, 526.4471f, 465.9183f, 420.6959f, 129.4022f, 485.2063f, 220.1614f,
        425.5884f, 224.9686f, 545.1217f, 353.6407f, 238.2388f, 62.7213f, 312.0847f, 78.3060f, 1.2788f, 465.1168f, 76.8773f, 507.2295f,
        350.7072f, 420.0901f, 499.0819f, 482.8026f, 295.2295f, 457.2856f, 318.5988f, 464.6119f, 248.9387f, 366.2193f, 368.7308f, 464.4846f,
        266.4057f, -43.0988f, 411.9049f, 94.8485f, 365.3591f, 230.8355f, 381.3726f, 246.8133f, 213.6699f, 419.1429f, 302.9046f, 467.1919f,
        282.3146f, 326.7091f, 321.6300f, 338.5049f, 157.0835f, 271.7193f, 238.9818f, 413.4953f, -3.7474f, 97.9864f, 45.0004f, 165.3309f,
        28.3577f, 158.4742f, 71.5941f, 260.1006f, 284.2465f, 120.1271f, 370.7495f, 246.4540f, 483.6205f, 186.3921f, 511.9348f, 335.0511f,
        -27.5488f, 218.5612f, 43.3521f, 243.6668f, 229.8062f, 103.3855f, 327.7773f, 223.5129f, 365.4548f, 86.1273f, 385.5540f, 219.3533f,
        343.5581f, 121.2852f, 483.2167f, 129.5677f, 234.4260f, 125.8439f, 310.7789f, 239.2034f, 248.4032f, 48.0437f, 371.5128f, 101.8978f,
        299.1465f, 387.2317f, 397.5784f, 484.8726f, 376.0880f, 262.2631f, 482.8782f, 339.8563f, 7.2930f, 47.0424f, 114.9965f, 86.7440f,
        397.3961f, 336.3557f, 528.7860f, 357.5037f, -33.2049f, 414.6207f, 59.2223f, 433.0458f, 396.8727f, 110.5703f, 439.3271f, 126.9654f,
        30.4567f, 27.2849f, 46.2837f, 123.3157f, 51.6484f, -22.3715f, 142.9798f, 30.9887f, -3.4962f, 6.9860f, 7.3904f, 40.2644f,
        204.1520f, 329.0802f, 241.1047f, 433.1711f, 162.1569f, 441.9229f, 172.2023f, 545.2635f, 41.6043f, -18.2279f, 124.3886f, 63.1082f,
        213.0999f, 303.8811f, 237.9903f, 444.1898f, 155.2101f, 6.7177f, 247.1608f, 65.1444f, 324.4111f, 233.2946f, 443.2500f, 358.8382f,
        384.8351f, 371.9398f, 508.2953f, 384.1355f, 302.7226f, 123.9848f, 349.8446f, 235.2196f, 20.8081f, -68.6720f, 103.6023f, 79.6067f,
        105.2511f, 234.0231f, 190.1397f, 361.1662f, 420.9290f, 451.9373f, 492.3893f, 539.3073f, -4.9387f, 81.6146f, 93.6732f, 176.0028f,
        187.2764f, 67.9256f, 219.5794f, 121.5657f, 397.7987f, 10.8413f, 544.7059f, 113.0846f, 467.5255f, 219.7334f, 483.1394f, 335.5223f,
        143.3246f, 223.3545f, 267.8786f, 373.0906f, 288.9383f, 358.9469f, 378.4586f, 433.3239f, 209.6311f, 371.4695f, 247.1145f, 381.6038f,
        320.6775f, 401.3793f, 432.7831f, 491.1622f, 8.9968f, 393.5190f, 22.5845f, 412.2537f, 13.8844f, 104.8985f, 130.2727f, 142.3685f,
        262.6455f, 252.9446f, 351.5533f, 302.9328f, 107.5252f, 93.7443f, 125.0270f, 203.6677f, 326.6030f, 150.6990f, 339.4493f, 179.0864f,
        119.1742f, 453.1236f, 232.0488f, 478.8208f, 420.9991f, 337.0981f, 465.6465f, 344.7978f, 342.8767f, 421.7388f, 476.3827f, 552.8516f,
        189.1445f, 156.2901f, 303.6933f, 260.6224f, 333.9324f, 265.2428f, 438.9627f, 272.1948f, 114.3128f, 240.9499f, 156.8251f, 246.1655f,
        193.8135f, 11.5223f, 300.4463f, 95.7648f, 27.6040f, 96.8022f, 169.8780f, 139.8998f, 423.1219f, 218.8621f, 437.7643f, 308.7743f,
        386.7347f, 0.8091f, 436.3329f, 66.5652f, 433.0917f, 396.4442f, 469.0579f, 535.0178f, 408.9413f, 39.9801f, 468.5356f, 83.8636f,
        423.9944f, 47.8940f, 535.6019f, 150.0867f, 78.3370f, 378.1336f, 149.9992f, 387.1877f, 422.8927f, -23.2443f, 508.9316f, 120.1789f,
        261.7021f, 376.5726f, 309.5111f, 523.7055f, 200.2215f, 307.2894f, 222.2736f, 418.4116f, 259.8004f, -0.8479f, 300.5735f, 69.4688f,
        106.7550f, 329.0340f, 235.8474f, 362.8130f, 98.8964f, 254.7818f, 189.6566f, 376.8467f, 91.9970f, 323.3163f, 149.3173f, 434.0331f,
        -18.1340f, 397.0634f, 100.5620f, 431.1345f, 242.9804f, 325.0598f, 253.5845f, 393.2908f, 424.4659f, 258.1096f, 463.2957f, 328.0667f,
        297.4333f, 99.1641f, 332.7187f, 223.2992f, 186.5782f, 297.1904f, 334.3975f, 400.0833f, 161.1921f, 430.0698f, 267.4008f, 526.9018f,
        185.6758f, 244.8488f, 278.7259f, 342.6730f, 103.7673f, 311.5224f, 105.5101f, 352.8224f, 397.2368f, 190.3715f, 425.6990f, 246.7565f,
        51.3437f, 374.1586f, 147.0393f, 381.9622f, 329.5223f, 439.7066f, 387.1005f, 557.9608f, 310.6336f, 47.4363f, 449.3514f, 112.9530f,
        229.9626f, 68.0539f, 344.9065f, 134.3514f, 397.6331f, 250.9398f, 465.2933f, 288.4979f, 89.1863f, 224.5854f, 201.8640f, 256.7900f,
        367.6410f, 241.4922f, 513.9763f, 330.0776f, 329.8622f, 6.7118f, 399.5483f, 42.3622f, 351.0067f, 196.8547f, 447.7431f, 207.4218f,
        263.3493f, 233.8098f, 401.2304f, 349.1684f, 404.1452f, 264.0487f, 442.1978f, 321.1426f, 430.0009f, 299.8394f, 563.0980f, 357.4945f,
        202.3143f, 327.4748f, 217.8485f, 392.7412f, 358.1485f, 259.5528f, 455.7672f, 381.9944f, 313.4684f, 370.7192f, 431.1113f, 419.5239f,
        180.1469f, 255.4066f, 272.7232f, 369.3540f, 426.0572f, 198.2577f, 500.8918f, 339.2499f, 150.7206f, 253.3635f, 243.7053f, 352.8329f,
        270.9340f, 17.9364f, 294.5319f, 83.2569f, 36.4112f, 80.3679f, 69.5312f, 192.7886f, 92.2801f, 229.0865f, 133.4951f, 298.3132f,
        375.3135f, 405.1188f, 465.3827f, 467.8684f, 164.8547f, 299.8922f, 231.6980f, 379.1594f, 178.3286f, 21.0337f, 215.7555f, 69.3744f,
        56.7212f, 287.8708f, 189.2598f, 304.4041f, 217.4480f, 79.4625f, 274.1624f, 142.2755f, 369.1791f, 357.2809f, 436.6378f, 376.7356f,
        416.5593f, 382.6425f, 478.6048f, 444.7983f, 21.0025f, 254.7366f, 49.1120f, 338.7197f, 232.4042f, 225.8433f, 342.4166f, 365.5193f,
        199.7265f, 166.0972f, 267.5468f, 172.4943f, 305.4298f, 176.3264f, 308.8521f, 269.9237f, 151.3188f, 397.4529f, 295.9569f, 466.6555f,
        138.0480f, 359.6507f, 260.5968f, 363.6696f, 181.5352f, 240.7855f, 290.3455f, 278.9682f, 225.7522f, 174.7890f, 356.2469f, 193.4433f,
        182.4345f, 8.5387f, 318.5487f, 41.8410f, 210.4292f, 50.5482f, 261.7152f, 92.4592f, 362.9012f, 66.1153f, 454.9341f, 126.9099f,
        326.9678f, 146.7783f, 418.6802f, 226.6052f, 150.2754f, 471.4981f, 191.1031f, 472.6456f, 383.2531f, 240.0174f, 417.3240f, 265.1360f,
        417.8392f, 109.9494f, 435.8114f, 124.8908f, 27.1272f, 11.4244f, 126.3650f, 94.3257f, 232.6628f, 144.1367f, 350.0197f, 194.1688f,
        85.4650f, 366.5097f, 199.8470f, 449.2209f, 345.5237f, 174.6456f, 393.6487f, 208.6972f, 103.6008f, 383.9478f, 135.1845f, 388.5580f,
        301.4075f, 330.7206f, 369.9960f, 471.9843f, 86.3247f, 46.8414f, 168.7999f, 63.9793f, 186.5999f, 294.3789f, 324.5439f, 314.2809f,
        408.6489f, 468.1303f, 539.9976f, 490.9658f, 121.9074f, 127.4639f, 259.4001f, 274.6741f, 374.0247f, -21.0436f, 501.7138f, 71.9877f,
        421.1110f, 415.6848f, 565.8336f, 507.6180f, 402.2457f, 367.8241f, 472.6052f, 515.8422f, 78.8962f, 253.9820f, 86.9698f, 268.1594f,
        403.1037f, 203.0262f, 416.5545f, 349.2269f, -13.5009f, 90.1716f, 45.6503f, 121.5695f, 176.9532f, 362.8065f, 216.3486f, 456.6442f,
        422.2061f, 217.5038f, 448.5273f, 281.0963f, 272.8624f, -12.1655f, 415.8898f, 46.0433f, 251.3114f, 271.6299f, 281.4290f, 411.3851f,
        121.9583f, 463.6307f, 265.9058f, 486.8656f, 348.9660f, 339.7936f, 463.3310f, 489.3569f, 306.5287f, 109.8543f, 403.0297f, 167.3439f,
        183.3392f, -22.1712f, 285.0661f, 75.4963f, 421.0473f, 397.5667f, 471.4370f, 542.7847f, 66.3152f, 463.7401f, 163.6328f, 473.3226f,
        70.7872f, 196.9543f, 99.6043f, 335.4611f, 251.0428f, 278.3568f, 391.7609f, 363.9607f, 463.0136f, 178.3225f, 508.9808f, 284.2776f,
        104.1169f, 198.2685f, 143.1397f, 221.4969f, 71.3536f, 19.4869f, 178.3168f, 99.9616f, 20.3440f, -2.3003f, 119.1549f, 99.0532f,
        396.1600f, 81.8756f, 464.4035f, 150.8565f, 65.5815f, 406.2740f, 160.8160f, 430.3668f, 239.2070f, 54.2293f, 263.9715f, 91.6030f,
        444.7733f, 49.1971f, 546.0992f, 177.5016f, -14.5900f, 271.2390f, 26.7309f, 277.3751f, 257.4168f, 54.2554f, 299.0693f, 160.8758f,
        243.5621f, 6.6488f, 268.7269f, 156.5579f, 378.4616f, 280.6006f, 428.9858f, 282.7156f, 152.4626f, 171.5487f, 202.8190f, 196.5445f,
        170.8344f, 262.3559f, 239.5070f, 363.8034f, 69.2827f, 451.1334f, 98.6552f, 461.0720f, 355.5286f, 31.0572f, 385.2867f, 119.9359f,
        351.4949f, 405.2588f, 433.2140f, 508.1748f, 58.2303f, 406.9281f, 78.4330f, 495.5619f, 144.9057f, 386.8375f, 248.5514f, 442.2501f,
        375.6284f, 263.1954f, 517.2766f, 368.0905f, -30.9426f, 265.2984f, 33.6499f, 354.8483f, 81.7472f, 303.6374f, 217.0119f, 335.5753f,
        269.6966f, 302.7942f, 285.3457f, 387.7014f, 163.3466f, -57.9610f, 170.7473f, 74.4432f, 81.7806f, 428.8672f, 190.2646f, 529.2253f,
        172.8226f, 257.1534f, 287.2148f, 328.4503f, 27.4537f, 366.2749f, 154.0694f, 415.1909f, 260.0797f, 181.7424f, 269.5455f, 195.5394f,
        294.9684f, -12.5261f, 411.7275f, 24.9233f, 259.0953f, 253.5339f, 316.1996f, 256.2007f, 23.4560f, 179.5914f, 69.6533f, 327.5987f,
        408.8140f, 201.4197f, 435.5946f, 235.5696f, 12.7857f, 108.6503f, 162.1921f, 231.0668f, 377.1631f, 111.8490f, 387.6489f, 137.9771f,
        118.1705f, 242.1441f, 242.3947f, 285.4007f, 343.2383f, 155.9774f, 439.5230f, 219.3007f, 47.8730f, 460.2977f, 158.3999f, 509.6342f,
        39.8081f, 26.4865f, 146.8540f, 146.4408f, 184.0596f, 87.9846f, 312.9663f, 231.6809f, 2.2755f, 81.2708f, 30.6605f, 212.6897f,
        112.0872f, 259.7130f, 113.2101f, 283.5961f, 316.9157f, 191.2768f, 407.0965f, 308.0034f, 391.8293f, 310.3482f, 445.5542f, 333.3923f,
        30.6705f, 406.4540f, 50.1148f, 543.5478f, 426.6715f, 103.5286f, 455.4062f, 181.6925f, 373.5433f, 320.8254f, 423.9739f, 371.9462f,
        429.1098f, 0.3217f, 440.5745f, 24.7185f, 344.4742f, 129.8145f, 353.9543f, 132.5740f, 268.3326f, 212.8878f, 405.8205f, 250.8319f,
        238.7950f, -53.0971f, 286.2983f, 84.0919f};
    std::vector<float> scores_vec = {
        0.9822f, 0.9644f, 0.1426f, 0.7149f, 0.6008f, 0.6906f, 0.0962f, 0.1886f, 0.0766f, 0.6041f, 0.9866f, 0.6720f,
        0.7108f, 0.9846f, 0.6780f, 0.0402f, 0.8670f, 0.3647f, 0.0044f, 0.5072f, 0.9370f, 0.2573f, 0.4915f, 0.1738f,
        0.0577f, 0.0805f, 0.7270f, 0.8641f, 0.1433f, 0.2883f, 0.1950f, 0.0269f, 0.5534f, 0.6999f, 0.6479f, 0.3881f,
        0.5550f, 0.0941f, 0.1543f, 0.9318f, 0.7615f, 0.9227f, 0.9167f, 0.6494f, 0.9282f, 0.4167f, 0.0036f, 0.0626f,
        0.1095f, 0.0954f, 0.3517f, 0.7013f, 0.7906f, 0.5902f, 0.1464f, 0.7479f, 0.3548f, 0.0130f, 0.2806f, 0.3306f,
        0.2742f, 0.8119f, 0.7599f, 0.6956f, 0.1390f, 0.8078f, 0.6772f, 0.1948f, 0.6481f, 0.4835f, 0.4394f, 0.1121f,
        0.5183f, 0.0999f, 0.1643f, 0.1325f, 0.9541f, 0.2849f, 0.3552f, 0.3221f, 0.8983f, 0.5630f, 0.9192f, 0.2999f,
        0.1148f, 0.5562f, 0.3455f, 0.8019f, 0.8794f, 0.4726f, 0.9714f, 0.5530f, 0.2709f, 0.4890f, 0.0373f, 0.8040f,
        0.1014f, 0.3087f, 0.5653f, 0.0430f, 0.0793f, 0.6961f, 0.0718f, 0.4771f, 0.3387f, 0.2281f, 0.1888f, 0.7634f,
        0.9515f, 0.1402f, 0.9597f, 0.5948f, 0.6417f, 0.7099f, 0.7041f, 0.8198f, 0.4835f, 0.5334f, 0.3238f, 0.1053f,
        0.6646f, 0.0336f, 0.2756f, 0.0942f, 0.1907f, 0.6387f, 0.6285f, 0.4211f, 0.0902f, 0.4334f, 0.3527f, 0.7205f,
        0.5790f, 0.4916f, 0.4870f, 0.9663f, 0.7563f, 0.4970f, 0.4792f, 0.0265f, 0.9425f, 0.3192f, 0.2559f, 0.9994f,
        0.7187f, 0.0474f, 0.0619f, 0.0255f, 0.5996f, 0.0716f, 0.9334f, 0.9369f, 0.5461f, 0.6166f, 0.2919f, 0.0640f,
        0.7375f, 0.1018f, 0.0856f, 0.3112f, 0.0125f, 0.4340f, 0.7077f, 0.8013f, 0.6043f, 0.8469f, 0.4065f, 0.8488f,
        0.5065f, 0.2230f, 0.9441f, 0.2750f, 0.0262f, 0.2427f, 0.3667f, 0.3513f, 0.5247f, 0.8831f, 0.2923f, 0.5208f,
        0.3401f, 0.8218f, 0.1576f, 0.1035f, 0.5030f, 0.6719f, 0.7955f, 0.5896f, 0.7738f, 0.3927f, 0.0329f, 0.1161f,
        0.0387f, 0.3289f, 0.4955f, 0.3563f, 0.5606f, 0.4806f, 0.6779f, 0.6670f, 0.3181f, 0.3462f, 0.5851f, 0.5964f,
        0.3147f, 0.3303f, 0.6940f, 0.6474f, 0.1351f, 0.4410f, 0.8927f, 0.0363f, 0.8552f, 0.1632f, 0.5072f, 0.4243f,
        0.0101f, 0.9154f, 0.4549f, 0.9543f, 0.2867f, 0.8663f, 0.9224f, 0.5568f, 0.2027f, 0.6852f, 0.5490f, 0.9445f,
        0.4393f, 0.2685f, 0.1383f, 0.6986f, 0.9741f, 0.0283f, 0.7404f, 0.9269f, 0.0748f, 0.1102f, 0.6920f, 0.6480f,
        0.0688f, 0.8344f, 0.5234f, 0.9072f, 0.8780f, 0.8125f, 0.5159f, 0.2517f, 0.5060f, 0.1008f, 0.6588f, 0.1340f,
        0.5112f, 0.0544f, 0.2995f, 0.2321f, 0.6200f, 0.7868f, 0.0573f, 0.8503f, 0.8608f, 0.3423f, 0.6590f, 0.4026f,
        0.1542f, 0.5287f, 0.0864f, 0.8785f, 0.9243f, 0.8216f, 0.5625f, 0.5576f, 0.9846f, 0.2479f, 0.0759f, 0.5619f,
        0.3288f, 0.3223f, 0.0071f, 0.5962f, 0.2640f, 0.1879f, 0.0404f, 0.3644f, 0.8790f, 0.3367f, 0.6791f, 0.7565f,
        0.3281f, 0.8216f, 0.6919f, 0.5592f, 0.0010f, 0.0351f, 0.9909f, 0.7823f, 0.9376f, 0.9023f, 0.0204f, 0.7918f,
        0.4511f, 0.7896f, 0.0067f, 0.2882f, 0.7513f, 0.7930f, 0.6197f, 0.3013f, 0.3104f, 0.9668f, 0.4392f, 0.4471f,
        0.5523f, 0.4095f, 0.5527f, 0.4323f, 0.8267f, 0.9091f, 0.9321f, 0.5643f, 0.4421f, 0.7052f, 0.8383f, 0.5630f,
        0.7000f, 0.7497f, 0.6764f, 0.7461f, 0.2086f, 0.4984f, 0.5883f, 0.0025f, 0.8560f, 0.6100f, 0.1291f, 0.8164f,
        0.7171f, 0.7583f, 0.3920f, 0.8542f, 0.4140f, 0.5705f, 0.0006f, 0.6449f, 0.7182f, 0.5671f, 0.4966f, 0.8099f,
        0.6814f, 0.2781f, 0.9591f, 0.7073f, 0.9879f, 0.9713f, 0.9189f, 0.7554f, 0.6094f, 0.1722f, 0.5434f, 0.7654f,
        0.5209f, 0.8682f, 0.1097f, 0.3809f, 0.5060f, 0.4323f, 0.1086f, 0.1535f, 0.8376f, 0.4844f, 0.0487f, 0.0165f,
        0.4735f, 0.1644f, 0.7051f, 0.7953f, 0.2283f, 0.5922f, 0.1544f, 0.3036f, 0.8888f, 0.5441f, 0.8859f, 0.2252f,
        0.3300f, 0.4710f, 0.4801f, 0.9976f, 0.1144f, 0.8520f, 0.8637f, 0.5532f, 0.3440f, 0.5192f, 0.2925f, 0.7991f,
        0.4983f, 0.9258f, 0.6227f, 0.5143f, 0.7111f, 0.5039f, 0.9045f, 0.1844f, 0.9733f, 0.8122f, 0.8607f, 0.4829f,
        0.8372f, 0.3068f, 0.7619f, 0.1405f, 0.3071f, 0.4457f, 0.3223f, 0.3870f, 0.8201f, 0.2567f, 0.7453f, 0.0737f,
        0.7657f, 0.7920f, 0.4017f, 0.7225f, 0.9151f, 0.8007f, 0.3904f, 0.4842f, 0.7794f, 0.2926f, 0.8039f, 0.3281f,
        0.8060f, 0.0868f, 0.0444f, 0.9977f, 0.8695f, 0.8828f, 0.9513f, 0.4383f, 0.2868f, 0.1300f, 0.5012f, 0.2200f,
        0.9356f, 0.0040f, 0.1432f, 0.2465f, 0.1990f, 0.2258f, 0.6560f, 0.3275f, 0.6150f, 0.8903f, 0.6026f, 0.6945f,
        0.3655f, 0.1597f, 0.3206f, 0.9643f, 0.6218f, 0.2775f, 0.4509f, 0.8355f, 0.6684f, 0.5607f, 0.8852f, 0.6724f,
        0.6427f, 0.1898f, 0.1064f, 0.9651f, 0.5989f, 0.4157f, 0.5890f, 0.0618f, 0.8221f, 0.2166f, 0.8045f, 0.5344f,
        0.2766f, 0.0302f, 0.8158f, 0.1765f, 0.0518f, 0.7559f, 0.3500f, 0.3893f, 0.2471f, 0.8592f, 0.2973f, 0.2102f,
        0.3092f, 0.2031f, 0.3177f, 0.0829f, 0.1585f, 0.4171f, 0.8795f, 0.0573f, 0.2127f, 0.9083f, 0.8900f, 0.6795f,
        0.2405f, 0.4198f, 0.2112f, 0.1286f, 0.3800f, 0.5758f, 0.3599f, 0.6108f, 0.2963f, 0.3459f, 0.7907f, 0.8783f,
        0.3220f, 0.5715f, 0.2782f, 0.0533f, 0.7379f, 0.1710f, 0.4257f, 0.4870f, 0.1845f, 0.0946f, 0.3480f, 0.9523f,
        0.6151f, 0.3814f, 0.0389f, 0.6003f, 0.0923f, 0.5425f, 0.7520f, 0.4236f, 0.2994f, 0.0474f, 0.0248f, 0.4300f,
        0.8833f, 0.2441f, 0.5741f, 0.6843f, 0.0608f, 0.1531f, 0.3313f, 0.6701f, 0.4390f, 0.7342f, 0.8676f, 0.7584f,
        0.9922f, 0.7544f, 0.8522f, 0.8324f, 0.7303f, 0.8018f, 0.9347f, 0.4752f, 0.6383f, 0.5149f, 0.8510f, 0.4314f,
        0.8197f, 0.7994f, 0.9619f, 0.2489f, 0.7096f, 0.7569f, 0.9363f, 0.9069f, 0.5735f, 0.5792f, 0.1673f, 0.9750f,
        0.2550f, 0.7247f, 0.7958f, 0.4412f, 0.2112f, 0.1890f, 0.8565f, 0.5108f, 0.0901f, 0.7170f, 0.2502f, 0.8764f,
        0.3096f, 0.2003f, 0.0849f, 0.5115f, 0.4507f, 0.7513f, 0.4646f, 0.3438f, 0.2617f, 0.2781f, 0.9278f, 0.1651f,
        0.9882f, 0.3269f, 0.0884f, 0.2487f, 0.0584f, 0.7900f, 0.5126f, 0.3370f, 0.6620f, 0.6306f, 0.9399f, 0.9613f,
        0.6807f, 0.8178f, 0.7924f, 0.4913f, 0.7045f, 0.0783f, 0.7580f, 0.9618f, 0.0850f, 0.8361f, 0.9330f, 0.2262f,
        0.5248f, 0.9279f, 0.9602f, 0.1279f, 0.3490f, 0.6981f, 0.2216f, 0.3248f, 0.0233f, 0.1535f, 0.5623f, 0.6531f,
        0.6489f, 0.7784f, 0.4153f, 0.2735f, 0.0156f, 0.2066f, 0.3124f, 0.1782f, 0.0201f, 0.1574f, 0.6661f, 0.6296f,
        0.9357f, 0.7982f, 0.5678f, 0.1376f, 0.5641f, 0.0616f, 0.4309f, 0.3903f, 0.4278f, 0.2798f, 0.6858f, 0.8409f,
        0.7685f, 0.6278f, 0.5383f, 0.0311f, 0.7229f, 0.5450f, 0.2707f, 0.3278f, 0.9356f, 0.6244f, 0.4759f, 0.6209f,
        0.4137f, 0.4702f, 0.2903f, 0.4399f, 0.6856f, 0.0399f, 0.7950f, 0.2830f, 0.6826f, 0.6427f, 0.6526f, 0.6081f,
        0.9591f, 0.5083f, 0.7323f, 0.7054f, 0.2363f, 0.2833f, 0.4240f, 0.2777f, 0.3667f, 0.3910f, 0.6039f, 0.2199f,
        0.8043f, 0.4375f, 0.7062f, 0.0814f, 0.4700f, 0.0282f, 0.6759f, 0.3437f, 0.9493f, 0.3241f, 0.5638f, 0.2574f,
        0.6201f, 0.4670f, 0.3706f, 0.2037f, 0.1115f, 0.1199f, 0.9990f, 0.4123f, 0.0019f, 0.9529f, 0.0200f, 0.4186f,
        0.7175f, 0.9146f, 0.7129f, 0.4636f, 0.9744f, 0.0393f, 0.9869f, 0.8494f, 0.9289f, 0.2548f, 0.1425f, 0.6633f,
        0.5159f, 0.5232f, 0.9246f, 0.6201f, 0.3111f, 0.4001f, 0.1335f, 0.1923f, 0.1434f, 0.8103f, 0.7049f, 0.5303f,
        0.3744f, 0.6685f, 0.8129f, 0.8812f, 0.5470f, 0.8199f, 0.5113f, 0.4745f, 0.8654f, 0.3864f, 0.3959f, 0.3049f,
        0.5187f, 0.5449f, 0.6605f, 0.4305f, 0.2178f, 0.8668f, 0.3460f, 0.9229f, 0.2074f, 0.5601f, 0.5366f, 0.8286f,
        0.1389f, 0.9099f, 0.5314f, 0.5861f, 0.5102f, 0.0360f, 0.4971f, 0.2635f, 0.3427f, 0.6491f, 0.4977f, 0.0932f,
        0.0730f, 0.1857f, 0.1909f, 0.6083f, 0.1778f, 0.8817f, 0.2098f, 0.0911f, 0.8757f, 0.2953f, 0.4254f, 0.9590f,
        0.9444f, 0.7149f, 0.0689f, 0.5933f, 0.9891f, 0.9469f, 0.1060f, 0.3960f};

    migraphx::parameter_map host_params;
    host_params["boxes"]  = migraphx::argument(boxes_s, boxes_vec.data());
    host_params["scores"] = migraphx::argument(scores_s, scores_vec.data());

    auto [indices, num_selected] = run_gpu_nms(std::move(p), host_params);
    indices.resize(static_cast<std::size_t>(num_selected) * 3);
    std::vector<int64_t> gold = {0, 0, 143, 0, 0, 10, 0, 0, 13, 0, 0, 0, 0, 0, 90, 0, 0, 135, 0, 0, 1, 0, 0, 76, 0, 0, 108, 0, 0, 170, 0, 0, 140, 0, 0, 20, 0, 0, 151, 0, 0, 150, 0, 0, 39, 0, 0, 44, 0, 0, 41, 0, 0, 82, 0, 0, 80, 0, 0, 88, 0, 0, 16, 0, 0, 27, 0, 0, 167, 0, 0, 165, 0, 0, 181, 0, 1, 187, 0, 1, 94, 0, 1, 152, 0, 1, 72, 0, 1, 32, 0, 1, 153, 0, 1, 109, 0, 1, 150, 0, 1, 19, 0, 1, 27, 0, 1, 96, 0, 1, 35, 0, 1, 197, 0, 1, 68, 0, 1, 22, 0, 1, 154, 0, 1, 17, 0, 1, 117, 0, 1, 43, 0, 1, 97, 0, 1, 10, 0, 1, 180, 0, 1, 182, 0, 1, 67, 0, 1, 44, 1, 0, 35, 1, 0, 152, 1, 0, 175, 1, 0, 4, 1, 0, 71, 1, 0, 166, 1, 0, 127, 1, 0, 38, 1, 0, 170, 1, 0, 44, 1, 0, 158, 1, 0, 198, 1, 0, 24, 1, 0, 101, 1, 0, 171, 1, 0, 2, 1, 0, 53, 1, 0, 102, 1, 0, 66, 1, 0, 140, 1, 0, 37, 1, 0, 98, 1, 0, 115, 1, 0, 150, 1, 0, 6, 1, 1, 114, 1, 1, 196, 1, 1, 0, 1, 1, 126, 1, 1, 124, 1, 1, 19, 1, 1, 11, 1, 1, 26, 1, 1, 84, 1, 1, 191, 1, 1, 117, 1, 1, 104, 1, 1, 197, 1, 1, 192, 1, 1, 10, 1, 1, 48, 1, 1, 68, 1, 1, 22, 1, 1, 128, 1, 1, 25, 1, 1, 134, 1, 1, 163, 1, 1, 121, 1, 1, 169, 1, 1, 185};
    EXPECT(migraphx::verify::verify_rms_range(indices, gold));
    EXPECT(num_selected == 100);
}


int main(int argc, const char* argv[]) { test::run(argc, argv); }
