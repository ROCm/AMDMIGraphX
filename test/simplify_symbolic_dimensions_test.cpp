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

#include <migraphx/dim_like.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/simplify_symbolic_dimensions.hpp>

#include "test.hpp"

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(m, {migraphx::simplify_symbolic_dimensions{}});
}

// A parameter-derived symbol can be the canonical expression, but the pass must preserve the
// parameter instruction and its externally visible shape.
TEST_CASE(simplify_symbolic_dimensions_preserve_parameter)
{
    using migraphx::sym::var;

    const auto parameter_count = var("parameter_count", {1, 8});
    const auto internal_count  = var("internal_count", {1, 8});
    const migraphx::shape parameter_shape{
        migraphx::shape::int64_type,
        std::vector<migraphx::shape::dynamic_dimension>{{parameter_count}}};

    migraphx::module m;
    auto parameter = m.add_parameter("parameter", parameter_shape);
    auto start = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
    auto limit = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{8}}});
    auto delta = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{1}}});
    auto range = m.add_instruction(
        migraphx::make_op(
            "dynamic_range",
            {{"max_output", 8},
             {"output_dim",
              migraphx::to_value(migraphx::shape::dynamic_dimension{internal_count})}}),
        start,
        limit,
        delta);
    // Putting the parameter first makes its expression canonical for the equivalent internal
    // range dimension.
    auto add = m.add_instruction(migraphx::make_op("add"), parameter, range);
    m.add_return({add});
    auto instruction_count = std::distance(m.begin(), m.end());

    run_pass(m);

    EXPECT(m.get_parameter("parameter") == parameter);
    EXPECT(m.get_parameter_shape("parameter") == parameter_shape);
    EXPECT(range->get_shape() == parameter_shape);
    EXPECT(m.get_output_shapes().front() == parameter_shape);
    EXPECT(std::distance(m.begin(), m.end()) == instruction_count);
    auto output_dim = migraphx::from_value<migraphx::shape::dynamic_dimension>(
        range->get_operator().to_value().at("output_dim"));
    EXPECT(output_dim.sym_expr == parameter_count);

    auto once = m;
    run_pass(m);
    EXPECT(m == once);
}

// Equivalent dimensions produced inside the graph should also canonicalize symbolic attributes
// that describe the same runtime extent.
TEST_CASE(simplify_symbolic_dimensions_shape_attributes)
{
    using migraphx::sym::var;

    const auto canonical_count = var("canonical_count", {1, 8});
    const auto alternate_count = var("alternate_count", {1, 8});

    migraphx::module m;
    auto indices = m.add_parameter("indices", {migraphx::shape::int64_type, {1, 8}});
    auto start = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
    auto limit = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{8}}});
    auto delta = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{1}}});
    auto canonical = m.add_instruction(
        migraphx::make_op(
            "dynamic_range",
            {{"max_output", 8},
             {"output_dim",
              migraphx::to_value(migraphx::shape::dynamic_dimension{canonical_count})}}),
        start,
        limit,
        delta);
    auto alternate = m.add_instruction(
        migraphx::make_op(
            "dynamic_range",
            {{"max_output", 8},
             {"output_dim",
              migraphx::to_value(migraphx::shape::dynamic_dimension{alternate_count})}}),
        start,
        limit,
        delta);
    // This pointwise edge establishes canonical_count == alternate_count.
    auto add    = m.add_instruction(migraphx::make_op("add"), canonical, alternate);
    auto sliced = m.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {1}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(alternate_count)}}}),
        indices,
        start,
        limit);
    auto squeezed  = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), sliced);
    auto broadcast = m.add_instruction(
        migraphx::make_op("multibroadcast",
                          {{"out_dyn_dims",
                            migraphx::to_value(std::vector<migraphx::shape::dynamic_dimension>{
                                {alternate_count}})}}),
        delta,
        squeezed);
    m.add_return({add, broadcast});
    auto instruction_count = std::distance(m.begin(), m.end());

    run_pass(m);

    // Attribute rewrites happen in place and their new shapes propagate through the return.
    EXPECT(std::distance(m.begin(), m.end()) == instruction_count);
    EXPECT(sliced->name() == "dyn_slice");
    EXPECT(broadcast->name() == "multibroadcast");
    auto ends = migraphx::from_value<std::vector<migraphx::sym::expr>>(
        sliced->get_operator().to_value().at("ends"));
    EXPECT(ends.front() == canonical_count);
    auto dims = migraphx::from_value<std::vector<migraphx::shape::dynamic_dimension>>(
        broadcast->get_operator().to_value().at("out_dyn_dims"));
    EXPECT(dims.front().sym_expr == canonical_count);
    EXPECT(m.get_output_shapes().front().dyn_dims().front().sym_expr == canonical_count);
    EXPECT(m.get_output_shapes().back().dyn_dims().front().sym_expr == canonical_count);
}

// Exercise the remaining mapped operation families with one shared equivalence class. The
// handlers must replace operations in place while leaving unrelated parameter metadata intact.
TEST_CASE(simplify_symbolic_dimensions_operation_handlers)
{
    using migraphx::sym::var;

    const auto canonical_count = var("canonical_count", {1, 8});
    const auto alternate_count = var("alternate_count", {1, 8});
    const migraphx::shape alternate_shape{
        migraphx::shape::int64_type,
        std::vector<migraphx::shape::dynamic_dimension>{{alternate_count}}};

    migraphx::module m;
    auto start = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
    auto limit = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{8}}});
    auto delta = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{1}}});
    auto canonical = m.add_instruction(
        migraphx::make_op(
            "dynamic_range",
            {{"max_output", 8},
             {"output_dim",
              migraphx::to_value(migraphx::shape::dynamic_dimension{canonical_count})}}),
        start,
        limit,
        delta);
    auto alternate = m.add_instruction(
        migraphx::make_op(
            "dynamic_range",
            {{"max_output", 8},
             {"output_dim",
              migraphx::to_value(migraphx::shape::dynamic_dimension{alternate_count})}}),
        start,
        limit,
        delta);
    // The first input chooses canonical_count for every alternate_count attribute below.
    auto add        = m.add_instruction(migraphx::make_op("add"), canonical, alternate);
    auto allocation = m.add_instruction(
        migraphx::make_op("allocate", {{"shape", migraphx::to_value(alternate_shape)}}));
    auto eval = m.add_instruction(
        migraphx::make_op(
            "eval_expr_from_shape",
            {{"expressions", migraphx::to_value(std::vector<migraphx::sym::expr>{alternate_count})},
             {"input_shapes", migraphx::to_value(std::vector<migraphx::shape>{alternate_shape})}}),
        alternate);
    auto reshape = m.add_instruction(
        migraphx::make_op("reshape",
                          {{"dims",
                            migraphx::to_value(std::vector<migraphx::dim_like>{
                                migraphx::shape::dynamic_dimension{alternate_count}})}}),
        alternate);
    m.add_return({add, allocation, canonical, alternate, eval, reshape});
    auto instruction_count = std::distance(m.begin(), m.end());

    run_pass(m);

    EXPECT(std::distance(m.begin(), m.end()) == instruction_count);

    auto allocation_shape =
        migraphx::from_value<migraphx::shape>(allocation->get_operator().to_value().at("shape"));
    EXPECT(allocation_shape.dyn_dims().front().sym_expr == canonical_count);

    auto output_dim = migraphx::from_value<migraphx::shape::dynamic_dimension>(
        alternate->get_operator().to_value().at("output_dim"));
    EXPECT(output_dim.sym_expr == canonical_count);
    EXPECT(alternate->get_shape().dyn_dims().front().sym_expr == canonical_count);

    const auto eval_attributes = eval->get_operator().to_value();
    auto expressions =
        migraphx::from_value<std::vector<migraphx::sym::expr>>(eval_attributes.at("expressions"));
    auto input_shapes =
        migraphx::from_value<std::vector<migraphx::shape>>(eval_attributes.at("input_shapes"));
    EXPECT(expressions.front() == canonical_count);
    EXPECT(input_shapes.front().dyn_dims().front().sym_expr == canonical_count);
    EXPECT(input_shapes.front().dyn_dims().back().sym_expr == canonical_count);

    auto dims = migraphx::from_value<std::vector<migraphx::dim_like>>(
        reshape->get_operator().to_value().at("dims"));
    EXPECT(std::get<migraphx::shape::dynamic_dimension>(dims.front()).sym_expr == canonical_count);
    EXPECT(reshape->get_shape().dyn_dims().front().sym_expr == canonical_count);
}

// Model the MaskRCNN post-processing pattern: NMS produces a runtime proposal count, TopK limits
// those proposals, and NonZero produces a separate count that later meets TopK at a pointwise op.
TEST_CASE(simplify_symbolic_dimensions_maskrcnn_postprocess)
{
    using migraphx::sym::lit;
    using migraphx::sym::resolve_min;
    using migraphx::sym::var;

    constexpr int64_t n_boxes = 6;
    constexpr int64_t k       = 4;
    const auto nms_count      = var("nms_count", {0, n_boxes});
    const auto nonzero_count  = var("nonzero_count", {0, k});

    migraphx::module m;
    auto boxes         = m.add_parameter("boxes", {migraphx::shape::float_type, {1, n_boxes, 4}});
    auto scores        = m.add_parameter("scores", {migraphx::shape::float_type, {1, 1, n_boxes}});
    auto max_output    = m.add_parameter("max_output", {migraphx::shape::int64_type, {1}});
    auto iou_threshold = m.add_parameter("iou_threshold", {migraphx::shape::float_type, {1}});
    auto score_threshold  = m.add_parameter("score_threshold", {migraphx::shape::float_type, {1}});
    auto mask             = m.add_parameter("mask", {migraphx::shape::bool_type, {2, 2}});
    auto parameters       = m.get_parameters();
    auto parameter_shapes = m.get_parameter_shapes();

    // Match the parser's NMS expansion: padded tuple outputs are trimmed to num_selected.
    auto nms =
        m.add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", true}}),
                          boxes,
                          scores,
                          max_output,
                          iou_threshold,
                          score_threshold);
    auto nms_indices  = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nms);
    auto num_selected = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nms);
    auto zero = m.add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {int64_t{0}}});
    auto nms_slice = m.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {0}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(nms_count)}}}),
        nms_indices,
        zero,
        num_selected);
    const auto nms_extent = nms_slice->get_shape().dyn_dims().front().sym_expr;
    auto box_column       = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {3}}}), nms_slice);
    auto box_indices = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), box_column);
    // Gather NMS-selected scores, then apply TopK so its output extent is min(k, nms_extent).
    auto flat_scores =
        m.add_instruction(migraphx::make_op("reshape", {{"dims", {n_boxes}}}), scores);
    auto gathered_scores =
        m.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), flat_scores, box_indices);
    auto topk = m.add_instruction(
        migraphx::make_op("topk", {{"k", k}, {"axis", 0}, {"largest", true}}), gathered_scores);
    auto topk_values = m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), topk);
    const auto topk_count = topk_values->get_shape().dyn_dims().front().sym_expr;

    // Match the parser's NonZero expansion and expose its runtime count as a rank-1 float tensor.
    auto nonzero = m.add_instruction(migraphx::make_op("nonzero"), mask);
    auto nonzero_indices =
        m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nonzero);
    auto num_nonzero =
        m.add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nonzero);
    auto nonzero_slice = m.add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {1}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(nonzero_count)}}}),
        nonzero_indices,
        zero,
        num_nonzero);
    auto nonzero_row = m.add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), nonzero_slice);
    nonzero_row = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), nonzero_row);
    auto nonzero_values = m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), nonzero_row);
    nonzero_values = m.add_instruction(
        migraphx::make_op("reshape",
                          {{"dims",
                            migraphx::to_value(std::vector<migraphx::dim_like>{
                                migraphx::shape::dynamic_dimension{nonzero_count}})}}),
        nonzero_values);

    // This is the symbolic collision: both inputs have the same runtime length but carry
    // independently created TopK and NonZero expressions.
    auto merged = m.add_instruction(migraphx::make_op("add"), topk_values, nonzero_values);
    m.add_return({merged, nms_slice, topk_values, nonzero_slice});
    auto instruction_count = std::distance(m.begin(), m.end());

    run_pass(m);

    // The pass only replaces mapped operations; module inputs and graph size stay unchanged.
    EXPECT(m.get_parameters() == parameters);
    EXPECT(m.get_parameter_shapes() == parameter_shapes);
    EXPECT(std::distance(m.begin(), m.end()) == instruction_count);
    EXPECT(nms->name() == "nonmaxsuppression");
    EXPECT(topk->name() == "topk");
    EXPECT(nonzero->name() == "nonzero");

    auto nms_ends = migraphx::from_value<std::vector<migraphx::sym::expr>>(
        nms_slice->get_operator().to_value().at("ends"));
    EXPECT(nms_ends.front() == nms_count);
    EXPECT(nms_extent == migraphx::sym::min(nms_count, lit(n_boxes)));
    EXPECT(nms_slice->get_shape().dyn_dims().front().sym_expr == nms_extent);
    EXPECT(topk_count == resolve_min(lit(k), nms_extent));
    EXPECT(topk_values->get_shape().dyn_dims().front().sym_expr == topk_count);

    // NonZero's count becomes the TopK expression selected by the first add input.
    auto nonzero_ends = migraphx::from_value<std::vector<migraphx::sym::expr>>(
        nonzero_slice->get_operator().to_value().at("ends"));
    EXPECT(nonzero_ends.front() == topk_count);
    EXPECT(nonzero_values->get_shape().dyn_dims().front().sym_expr == topk_count);
    EXPECT(merged->get_shape().dyn_dims().front().sym_expr == topk_count);
    EXPECT(m.get_output_shapes().front().dyn_dims().front().sym_expr == topk_count);

    auto once = m;
    run_pass(m);
    EXPECT(m == once);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
