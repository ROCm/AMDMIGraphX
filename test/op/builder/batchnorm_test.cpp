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

#include <op_builder_test_utils.hpp>

TEST_CASE(batchnorm_rank_0_op_builder_test)
{
    migraphx::module mm;

    mm.add_parameter("x", {migraphx::shape::half_type, {}});
    mm.add_parameter("scale", {migraphx::shape::float_type, {3}});
    mm.add_parameter("bias", {migraphx::shape::float_type, {3}});
    mm.add_parameter("mean", {migraphx::shape::float_type, {3}});
    mm.add_parameter("variance", {migraphx::shape::float_type, {3}});

    EXPECT(test::throws<migraphx::exception>(
        [&] { make_op_module("batchnorm", {}, mm.get_parameters()); },
        "rank 0 input tensor, unhandled data format"));
}

TEST_CASE(batchnorm_rank_1_op_builder_test)
{
    migraphx::module mm;

    const float epsilon = 1e-6f;

    auto x     = mm.add_parameter("x", {migraphx::shape::float_type, {10}});
    auto scale = mm.add_parameter("scale", {migraphx::shape::float_type, {1}});
    auto bias  = mm.add_parameter("bias", {migraphx::shape::float_type, {1}});
    auto mean  = mm.add_parameter("mean", {migraphx::shape::float_type, {1}});
    auto var   = mm.add_parameter("variance", {migraphx::shape::float_type, {1}});

    auto eps = mm.add_literal(migraphx::literal{migraphx::shape::float_type, {epsilon}});

    auto x_sub_mean = add_common_op(mm, migraphx::make_op("sub"), {x, mean});
    auto var_eps    = add_common_op(mm, migraphx::make_op("add"), {var, eps});
    auto rsqrt      = mm.add_instruction(migraphx::make_op("rsqrt"), {var_eps});
    auto mul0       = add_common_op(mm, migraphx::make_op("mul"), {scale, rsqrt});
    auto r0         = add_common_op(mm, migraphx::make_op("mul"), {x_sub_mean, mul0});
    add_common_op(mm, migraphx::make_op("add"), {r0, bias});

    EXPECT(mm == make_op_module("batchnorm", {{"epsilon", epsilon}}, mm.get_parameters()));
}

TEST_CASE(batchnorm_rank_larger_than_2_op_builder_test)
{
    migraphx::module mm;

    auto x     = mm.add_parameter("x", {migraphx::shape::half_type, {2, 3, 4}});
    auto scale = mm.add_parameter("scale", {migraphx::shape::float_type, {3}});
    auto bias  = mm.add_parameter("bias", {migraphx::shape::float_type, {3}});
    auto mean  = mm.add_parameter("mean", {migraphx::shape::float_type, {3}});
    auto var   = mm.add_parameter("variance", {migraphx::shape::float_type, {3}});

    auto eps = mm.add_literal(migraphx::literal{migraphx::shape::half_type, {1e-5f}});

    auto usq_scale = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), scale);
    auto usq_bias  = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), bias);
    auto usq_mean  = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), mean);
    auto usq_var   = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), var);

    auto x_sub_mean = add_common_op(mm, migraphx::make_op("sub"), {x, usq_mean});
    auto var_eps    = add_common_op(mm, migraphx::make_op("add"), {usq_var, eps});
    auto rsqrt      = mm.add_instruction(migraphx::make_op("rsqrt"), var_eps);
    auto mul0       = add_common_op(mm, migraphx::make_op("mul"), {usq_scale, rsqrt});
    auto r0         = add_common_op(mm, migraphx::make_op("mul"), {x_sub_mean, mul0});
    add_common_op(mm, migraphx::make_op("add"), {r0, usq_bias});

    EXPECT(mm == make_op_module("batchnorm", {}, mm.get_parameters()));
}

TEST_CASE(batchnorm_invalid_arguments_op_builder_test)
{
    migraphx::module mm;

    mm.add_parameter("x", {migraphx::shape::half_type, {2}});
    mm.add_parameter("scale", {migraphx::shape::float_type, {3, 2}});

    EXPECT(test::throws<migraphx::exception>(
        [&] { make_op_module("batchnorm", {}, mm.get_parameters()); },
        "argument scale, bias, mean, or var rank != 1"));
}
