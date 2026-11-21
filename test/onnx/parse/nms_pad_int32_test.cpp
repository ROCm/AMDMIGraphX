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

#include <onnx_test.hpp>

TEST_CASE(nms_pad_int32_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto start = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {2}});
    auto end   = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {3}});
    mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {1}});
    auto size = mm->add_literal(migraphx::literal{{migraphx::shape::int32_type, {1}}, {10}});
    auto zero =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type}, {0}});
    auto b   = mm->add_parameter("boxes", {migraphx::shape::float_type, {1, 6, 4}});
    auto s   = mm->add_parameter("scores", {migraphx::shape::float_type, {1, 1, 6}});
    auto mo  = mm->add_parameter("max_output_boxes_per_class", {migraphx::shape::int64_type, {1}});
    auto iou = mm->add_parameter("iou_threshold", {migraphx::shape::float_type, {1}});
    auto st  = mm->add_parameter("score_threshold", {migraphx::shape::float_type, {1}});

    auto nms     = mm->add_instruction(migraphx::make_op("nonmaxsuppression"), b, s, mo, iou, st);
    auto indices = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), nms);
    auto num_selected =
        mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), nms);
    // Added last so it lands at the front of the module, where the parser puts it
    auto starts   = mm->add_literal(migraphx::literal{{migraphx::shape::int64_type, {1}}, {0}});
    auto n        = var("main_NonMaxSuppression_10", {0, 6});
    auto selected = mm->add_instruction(
        migraphx::make_op("dyn_slice",
                          {{"axes", {0}},
                           {"starts", {0}},
                           {"ends", migraphx::value::array{migraphx::to_value(n)}}}),
        indices,
        starts,
        num_selected);
    auto column = mm->add_instruction(
        migraphx::make_op("dyn_slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {3}}}),
        selected,
        start,
        end);
    column           = mm->add_instruction(migraphx::make_op("contiguous"), column);
    auto box_indices = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), column);
    auto num = mm->add_instruction(migraphx::make_op("dimensions_of", {{"end", 1}}), box_indices);
    num      = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), num);
    auto pad = mm->add_instruction(migraphx::make_op("sub"), size, num);
    pad      = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int64_type}}), pad);
    // The pad size is the expression 10 - n, so the concat has the static size 10
    std::vector<migraphx::shape::dynamic_dimension> pad_dims{
        migraphx::shape::dynamic_dimension{lit(10) - n}};
    auto zeros = mm->add_instruction(
        migraphx::make_op("broadcast_with_dims", {{"out_dyn_dims", migraphx::to_value(pad_dims)}}),
        zero,
        pad);
    auto ret = mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), box_indices, zeros);
    mm->add_return({ret});

    auto prog = read_onnx("nms_pad_int32_test.onnx");
    EXPECT(p == prog);
}
