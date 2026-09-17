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

#include <cstdint>
#include <vector>
#include <op_builder_test_utils.hpp>
#include <migraphx/make_op.hpp>

// tm::group_norm reshapes to (N, num_groups, -1), normalizes over the trailing axis,
// reshapes back, then applies the per-channel affine.
TEST_CASE(torch_kit_group_norm_op_builder_test)
{
    const auto f    = migraphx::shape::float_type;
    const float eps = 1e-5f;

    migraphx::module mm;
    auto x     = mm.add_parameter("x", {f, {2, 4, 3}});
    auto scale = mm.add_parameter("scale", {f, {4}});
    auto bias  = mm.add_parameter("bias", {f, {4}});

    std::vector<int64_t> axes = {-1};
    auto grouped  = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, -1}}}), x);
    auto mean     = mm.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), grouped);
    auto x_sub    = add_common_op(mm, migraphx::make_op("sub"), {grouped, mean});
    auto sqdiff   = add_common_op(mm, migraphx::make_op("sqdiff"), {grouped, mean});
    auto variance = mm.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), sqdiff);
    auto eps_lit  = mm.add_literal(migraphx::literal{migraphx::shape{f}, {eps}});
    auto var_eps  = add_common_op(mm, migraphx::make_op("add"), {variance, eps_lit});
    auto rsqrt    = mm.add_instruction(migraphx::make_op("rsqrt"), var_eps);
    auto norm     = add_common_op(mm, migraphx::make_op("mul"), {x_sub, rsqrt});
    auto norm_r   = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 4, 3}}}), norm);
    auto scale_u  = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), scale);
    auto bias_u   = mm.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), bias);
    auto scaled   = add_common_op(mm, migraphx::make_op("mul"), {norm_r, scale_u});
    add_common_op(mm, migraphx::make_op("add"), {scaled, bias_u});

    EXPECT(mm == make_op_module(
                     "tm::group_norm", {{"epsilon", eps}, {"num_groups", 2}}, mm.get_parameters()));
}

// num_groups must divide the channel dim and the input must have spatial dims.
TEST_CASE(torch_kit_group_norm_bad_input_op_builder_test)
{
    const auto f = migraphx::shape::float_type;
    migraphx::module mm;
    mm.add_parameter("x", {f, {2, 3, 4}}); // 3 channels not divisible by num_groups = 2
    EXPECT(test::throws<migraphx::exception>([&] {
        make_op_module(
            "tm::group_norm", {{"epsilon", 1e-5f}, {"num_groups", 2}}, mm.get_parameters());
    }));
}
