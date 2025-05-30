/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <onnx_test_utils.hpp>

TEST_CASE(group_norm_contrib_channels_last_silu_3d_test)
{
    const std::vector<int64_t> input_dims{1, 4, 2};
    const std::vector<int64_t> scale_dims{2};
    const std::vector<int64_t> bias_dims{2};
    const std::vector<int64_t> reshape_dims{1, 2, 1, 4};
    const std::vector<int64_t> reduce_axes{2, 3};
    const float eps_value               = 1e-5f;
    const migraphx::shape::type_t dtype = migraphx::shape::float_type;

    migraphx::program p;
    auto* mm = p.get_main_module();

    auto x     = mm->add_parameter("x", {dtype, input_dims});
    auto scale = mm->add_parameter("gamma", {dtype, scale_dims});
    auto bias  = mm->add_parameter("beta", {dtype, bias_dims});

    auto eps = mm->add_literal(migraphx::literal{dtype, {eps_value}});

    auto x_transp =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), x);

    auto x_dims = x_transp->get_shape().lens();

    auto x_reshaped =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", reshape_dims}}), x_transp);
    auto mean =
        mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", reduce_axes}}), x_reshaped);
    auto x_sub_mean    = add_common_op(*mm, migraphx::make_op("sub"), {x_reshaped, mean});
    auto x_sqdiff_mean = add_common_op(*mm, migraphx::make_op("sqdiff"), {x_reshaped, mean});
    auto var     = mm->add_instruction(migraphx::make_op("reduce_mean", {{"axes", reduce_axes}}),
                                   x_sqdiff_mean);
    auto var_eps = add_common_op(*mm, migraphx::make_op("add"), {var, eps});
    auto rsqrt   = mm->add_instruction(migraphx::make_op("rsqrt"), {var_eps});
    auto result  = add_common_op(*mm, migraphx::make_op("mul"), {x_sub_mean, rsqrt});
    auto result_reshaped =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", x_dims}}), result);
    auto scale_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", x_dims}}), scale);
    auto bias_bcast = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", x_dims}}), bias);
    auto scaled = mm->add_instruction(migraphx::make_op("mul"), {result_reshaped, scale_bcast});
    auto y      = mm->add_instruction(migraphx::make_op("add"), {scaled, bias_bcast});
    auto group_out =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), y);

    // Order matters for this operator as we MUST have SILU after a group norm is done when the
    // activation flag is set
    auto sigmoid_out = mm->add_instruction(migraphx::make_op("sigmoid"), group_out);
    mm->add_instruction(migraphx::make_op("mul"), group_out, sigmoid_out);

    auto prog = optimize_onnx("group_norm_contrib_channels_last_and_silu_3d_test.onnx");
    EXPECT(p == prog);
}
