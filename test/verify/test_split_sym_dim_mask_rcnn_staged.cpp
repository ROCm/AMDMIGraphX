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

#include "verify_program.hpp"
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/program.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>

struct test_split_sym_dim_mask_rcnn_staged : verify_program<test_split_sym_dim_mask_rcnn_staged>
{
    migraphx::program create_program() const
    {
        using migraphx::sym::var;

        migraphx::program p;
        auto* mm                = p.get_main_module();
        auto x                  = mm->add_parameter("x", {migraphx::shape::float_type, {4}});
        auto nonempty_condition = mm->add_instruction(migraphx::make_op("equal"), x, x);
        auto empty_condition    = mm->add_instruction(migraphx::make_op("not"), nonempty_condition);
        auto starts = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});

        auto nonempty = mm->add_instruction(migraphx::make_op("nonzero"), nonempty_condition);
        auto nonempty_buffer =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nonempty);
        auto nonempty_count =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nonempty);
        auto proposal_indices = mm->add_instruction(
            migraphx::make_op(
                "dyn_slice",
                {{"axes", {1}},
                 {"starts", {0}},
                 {"ends",
                  migraphx::value::array{migraphx::to_value(var("mask_proposal_count", {0, 4}))}}}),
            nonempty_buffer,
            starts,
            nonempty_count);
        proposal_indices =
            mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), proposal_indices);

        auto boxes_table = mm->add_literal(migraphx::literal{{migraphx::shape::float_type, {4, 4}},
                                                             {0.0f,
                                                              0.0f,
                                                              1.0f,
                                                              1.0f,
                                                              0.1f,
                                                              0.1f,
                                                              1.1f,
                                                              1.1f,
                                                              2.0f,
                                                              2.0f,
                                                              3.0f,
                                                              3.0f,
                                                              4.0f,
                                                              4.0f,
                                                              5.0f,
                                                              5.0f}});
        auto scores_table0 = mm->add_literal(
            migraphx::literal{{migraphx::shape::float_type, {4}}, {0.9f, 0.8f, 0.7f, 0.6f}});
        auto scores_table1 = mm->add_literal(
            migraphx::literal{{migraphx::shape::float_type, {4}}, {0.6f, 0.7f, 0.8f, 0.9f}});

        auto boxes = mm->add_instruction(
            migraphx::make_op("gather", {{"axis", 0}}), boxes_table, proposal_indices);
        boxes        = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), boxes);
        auto scores0 = mm->add_instruction(
            migraphx::make_op("gather", {{"axis", 0}}), scores_table0, proposal_indices);
        scores0 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}), scores0);
        auto scores1 = mm->add_instruction(
            migraphx::make_op("gather", {{"axis", 0}}), scores_table1, proposal_indices);
        scores1 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 1}}}), scores1);

        auto max_output      = mm->add_literal(int64_t{3});
        auto iou_threshold   = mm->add_literal(0.5f);
        auto score_threshold = mm->add_literal(0.0f);
        auto nms0            = mm->add_instruction(migraphx::make_op("nonmaxsuppression"),
                                        boxes,
                                        scores0,
                                        max_output,
                                        iou_threshold,
                                        score_threshold);
        auto nms_buffer0 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nms0);
        auto nms_count0 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nms0);
        auto selected0 = mm->add_instruction(
            migraphx::make_op(
                "dyn_slice",
                {{"axes", {0}},
                 {"starts", {0}},
                 {"ends",
                  migraphx::value::array{migraphx::to_value(var("mask_nms_count0", {0, 4}))}}}),
            nms_buffer0,
            starts,
            nms_count0);

        auto nms1 = mm->add_instruction(migraphx::make_op("nonmaxsuppression"),
                                        boxes,
                                        scores1,
                                        max_output,
                                        iou_threshold,
                                        score_threshold);
        auto nms_buffer1 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nms1);
        auto nms_count1 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nms1);
        auto selected1 = mm->add_instruction(
            migraphx::make_op(
                "dyn_slice",
                {{"axes", {0}},
                 {"starts", {0}},
                 {"ends",
                  migraphx::value::array{migraphx::to_value(var("mask_nms_count1", {0, 4}))}}}),
            nms_buffer1,
            starts,
            nms_count1);

        auto nms_counts = mm->add_instruction(
            migraphx::make_op("eval_expr_from_shape",
                              {{"expressions",
                                migraphx::to_value(std::vector<migraphx::sym::expr>{
                                    var("mask_nms_count0"), var("mask_nms_count1")})}}),
            selected0,
            selected1);
        auto joined = mm->add_instruction(
            migraphx::make_op("dyn_concat", {{"axis", 0}}), nms_counts, selected0, selected1);
        auto joined_buffer =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), joined);
        auto joined_count =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), joined);
        auto merged = mm->add_instruction(
            migraphx::make_op(
                "dyn_slice",
                {{"axes", {0}},
                 {"starts", {0}},
                 {"ends",
                  migraphx::value::array{migraphx::to_value(var("mask_merged_count", {0, 8}))}}}),
            joined_buffer,
            starts,
            joined_count);

        auto empty = mm->add_instruction(migraphx::make_op("nonzero"), empty_condition);
        auto empty_buffer =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), empty);
        auto empty_count =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), empty);
        auto empty_indices = mm->add_instruction(
            migraphx::make_op(
                "dyn_slice",
                {{"axes", {1}},
                 {"starts", {0}},
                 {"ends",
                  migraphx::value::array{migraphx::to_value(var("mask_empty_count", {0, 4}))}}}),
            empty_buffer,
            starts,
            empty_count);

        mm->add_return({merged, empty_indices});
        return p;
    }
};
