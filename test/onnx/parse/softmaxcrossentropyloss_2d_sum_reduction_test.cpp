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

TEST_CASE(softmaxcrossentropyloss_2d_sum_reduction_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto scores = mm->add_parameter("0", migraphx::shape{migraphx::shape::float_type, {4, 4}});
    auto labels = mm->add_parameter("1", migraphx::shape{migraphx::shape::int32_type, {4}});
    mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::int64_type, {1}, {0}), {-1}));

    auto softmax       = mm->add_instruction(migraphx::make_op("softmax"), scores);
    auto logsoftmax    = mm->add_instruction(migraphx::make_op("log"), softmax);
    auto neglogsoftmax = mm->add_instruction(migraphx::make_op("neg"), logsoftmax);

    auto loss =
        mm->add_instruction(migraphx::make_op("gather", {{"axis", 1}}), neglogsoftmax, labels);
    loss = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), loss);
    mm->add_return({loss});

    auto prog = migraphx::parse_onnx("softmaxcrossentropyloss_2d_sum_reduction_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(softmaxcrossentropyloss_2d_sum_reduction_double_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto scores = mm->add_parameter("0", migraphx::shape{migraphx::shape::double_type, {4, 4}});
    auto labels = mm->add_parameter("1", migraphx::shape{migraphx::shape::int32_type, {4}});
    mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::int64_type, {1}, {0}), {-1}));

    auto softmax       = mm->add_instruction(migraphx::make_op("softmax"), scores);
    auto logsoftmax    = mm->add_instruction(migraphx::make_op("log"), softmax);
    auto neglogsoftmax = mm->add_instruction(migraphx::make_op("neg"), logsoftmax);

    auto loss =
        mm->add_instruction(migraphx::make_op("gather", {{"axis", 1}}), neglogsoftmax, labels);
    loss = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), loss);
    mm->add_return({loss});

    auto prog = migraphx::parse_onnx("softmaxcrossentropyloss_2d_sum_reduction_double_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(softmaxcrossentropyloss_2d_sum_reduction_half_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto scores = mm->add_parameter("0", migraphx::shape{migraphx::shape::half_type, {4, 4}});
    auto labels = mm->add_parameter("1", migraphx::shape{migraphx::shape::int32_type, {4}});
    mm->add_literal(
        migraphx::literal(migraphx::shape(migraphx::shape::int64_type, {1}, {0}), {-1}));

    auto softmax       = mm->add_instruction(migraphx::make_op("softmax"), scores);
    auto logsoftmax    = mm->add_instruction(migraphx::make_op("log"), softmax);
    auto neglogsoftmax = mm->add_instruction(migraphx::make_op("neg"), logsoftmax);

    auto loss =
        mm->add_instruction(migraphx::make_op("gather", {{"axis", 1}}), neglogsoftmax, labels);
    loss = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), loss);
    mm->add_return({loss});

    auto prog = migraphx::parse_onnx("softmaxcrossentropyloss_2d_sum_reduction_half_test.onnx");

    EXPECT(p == prog);
}
