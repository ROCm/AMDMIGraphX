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

TEST_CASE(convinteger_dual_bias_test)
{
    migraphx::program p;
    auto* mm         = p.get_main_module();
    auto data        = mm->add_parameter("0", {migraphx::shape::int8_type, {1, 3, 5, 5}});
    auto weight      = mm->add_parameter("1", {migraphx::shape::int8_type, {1, 3, 2, 2}});
    auto data_bias   = mm->add_parameter("2", {migraphx::shape::int8_type, {1}, {1}});
    auto weight_bias = mm->add_parameter("3", {migraphx::shape::int8_type, {1}, {1}});

    auto quant = mm->add_instruction(migraphx::make_op("quant_convolution"), data, weight);

    auto mbcast_data_bias = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", weight->get_shape().lens()}}), data_bias);

    auto quant_db_w =
        mm->add_instruction(migraphx::make_op("quant_convolution"), mbcast_data_bias, weight);

    auto quant_mb_w = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", quant->get_shape().lens()}}), quant_db_w);

    quant = mm->add_instruction(migraphx::make_op("sub"), quant, quant_mb_w);

    auto mbcast_weight_bias = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", data->get_shape().lens()}}), weight_bias);

    auto quant_d_wb =
        mm->add_instruction(migraphx::make_op("quant_convolution"), data, mbcast_weight_bias);

    auto quant_md_wb = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", quant->get_shape().lens()}}), quant_d_wb);

    quant = mm->add_instruction(migraphx::make_op("sub"), quant, quant_md_wb);

    auto bcast_data_bias = mm->add_instruction(
        migraphx::make_op("broadcast", {{"out_lens", data->get_shape().lens()}}), data_bias);
    auto bcast_weight_bias = mm->add_instruction(
        migraphx::make_op("broadcast", {{"out_lens", weight->get_shape().lens()}}), weight_bias);
    auto bias_quant = mm->add_instruction(
        migraphx::make_op("quant_convolution"), bcast_data_bias, bcast_weight_bias);

    mm->add_instruction(migraphx::make_op("add"), quant, bias_quant);

    auto prog = optimize_onnx("convinteger_dual_bias_test.onnx");
    EXPECT(p == prog);
}
