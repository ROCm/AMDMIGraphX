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

#include <onnx_test.hpp>

TEST_CASE(softmaxcrossentropyloss_kd_sum_reduction_weighted_double_test)
{
    migraphx::program p;
    auto* mm          = p.get_main_module();
    size_t batch_size = 4;
    size_t class_size = 4;

    auto scores = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::double_type, {batch_size, class_size, 2, 2}});
    auto labels =
        mm->add_parameter("1", migraphx::shape{migraphx::shape::int32_type, {batch_size, 2, 2}});
    auto weights =
        mm->add_parameter("2", migraphx::shape{migraphx::shape::double_type, {batch_size}});

    auto weight_tensor =
        mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), weights, labels);

    mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::int64_type, {1}, {0}), {-1}));

    auto rs_scores = mm->add_instruction(
        migraphx::make_op("reshape", {{"dims", {batch_size, class_size, 4}}}), scores);

    auto softmax       = mm->add_instruction(migraphx::make_op("softmax"), rs_scores);
    auto logsoftmax    = mm->add_instruction(migraphx::make_op("log"), softmax);
    auto neglogsoftmax = mm->add_instruction(migraphx::make_op("neg"), logsoftmax);

    auto loss =
        mm->add_instruction(migraphx::make_op("gather", {{"axis", 1}}), neglogsoftmax, labels);

    mm->debug_print();
    loss = mm->add_instruction(migraphx::make_op("reshape", {{"dims", labels->get_shape().lens()}}),
                               loss);

    loss = mm->add_instruction(migraphx::make_op("mul"), loss, weight_tensor);
    loss = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), loss);
    mm->add_return({loss});

    auto prog = migraphx::parse_onnx("softmaxcrossentropyloss_kd_sum_reduction_double_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(softmaxcrossentropyloss_kd_no_reduction_weighted_test)
{
    migraphx::program p;
    auto* mm          = p.get_main_module();
    size_t batch_size = 4;
    size_t class_size = 4;

    auto scores = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {batch_size, class_size, 2, 2}});
    auto labels =
        mm->add_parameter("1", migraphx::shape{migraphx::shape::int32_type, {batch_size, 2, 2}});
    auto weights =
        mm->add_parameter("2", migraphx::shape{migraphx::shape::float_type, {batch_size}});

    auto weight_tensor =
        mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), weights, labels);

    mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::int64_type, {1}, {0}), {-1}));

    auto rs_scores = mm->add_instruction(
        migraphx::make_op("reshape", {{"dims", {batch_size, class_size, 4}}}), scores);

    auto softmax       = mm->add_instruction(migraphx::make_op("softmax"), rs_scores);
    auto logsoftmax    = mm->add_instruction(migraphx::make_op("log"), softmax);
    auto neglogsoftmax = mm->add_instruction(migraphx::make_op("neg"), logsoftmax);

    auto loss =
        mm->add_instruction(migraphx::make_op("gather", {{"axis", 1}}), neglogsoftmax, labels);

    mm->debug_print();
    loss = mm->add_instruction(migraphx::make_op("reshape", {{"dims", labels->get_shape().lens()}}),
                               loss);

    loss = mm->add_instruction(migraphx::make_op("mul"), loss, weight_tensor);
    loss = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), loss);
    mm->add_return({loss});

    auto prog = migraphx::parse_onnx("softmaxcrossentropyloss_kd_no_reduction_weightd_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(softmaxcrossentropyloss_kd_mean_reduction_half_weighted_test)
{
    migraphx::program p;
    auto* mm          = p.get_main_module();
    size_t batch_size = 4;
    size_t class_size = 4;

    auto scores = mm->add_parameter(
        "0", migraphx::shape{migraphx::shape::half_type, {batch_size, class_size, 2, 2}});
    auto labels =
        mm->add_parameter("1", migraphx::shape{migraphx::shape::int32_type, {batch_size, 2, 2}});
    auto weights =
        mm->add_parameter("2", migraphx::shape{migraphx::shape::half_type, {batch_size}});

    auto weight_tensor =
        mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), weights, labels);

    mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::int64_type, {1}, {0}), {-1}));

    auto rs_scores = mm->add_instruction(
        migraphx::make_op("reshape", {{"dims", {batch_size, class_size, 4}}}), scores);

    auto softmax       = mm->add_instruction(migraphx::make_op("softmax"), rs_scores);
    auto logsoftmax    = mm->add_instruction(migraphx::make_op("log"), softmax);
    auto neglogsoftmax = mm->add_instruction(migraphx::make_op("neg"), logsoftmax);

    auto loss =
        mm->add_instruction(migraphx::make_op("gather", {{"axis", 1}}), neglogsoftmax, labels);

    mm->debug_print();
    loss = mm->add_instruction(migraphx::make_op("reshape", {{"dims", labels->get_shape().lens()}}),
                               loss);

    loss = mm->add_instruction(migraphx::make_op("mul"), loss, weight_tensor);
    loss = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), loss);
    mm->add_return({loss});

    auto prog =
        migraphx::parse_onnx("softmaxcrossentropyloss_kd_mean_reduction_half_weightd_test.onnx");

    EXPECT(p == prog);
}
