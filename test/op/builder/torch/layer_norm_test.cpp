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

// tm::layer_norm == (x - mean) * rsqrt(var + eps) * scale + bias, reduced over `axes`,
// with the affine params broadcast right-aligned against the input.
TEST_CASE(torch_kit_layer_norm_op_builder_test)
{
    const auto f              = migraphx::shape::float_type;
    const float eps           = 1e-5f;
    std::vector<int64_t> axes = {-1};

    migraphx::module mm;
    auto x        = mm.add_parameter("x", {f, {2, 3, 4}});
    auto scale    = mm.add_parameter("scale", {f, {4}});
    auto bias     = mm.add_parameter("bias", {f, {4}});
    auto mean     = mm.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), x);
    auto x_sub    = add_common_op(mm, migraphx::make_op("sub"), {x, mean});
    auto sqdiff   = add_common_op(mm, migraphx::make_op("sqdiff"), {x, mean});
    auto variance = mm.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), sqdiff);
    auto eps_lit  = mm.add_literal(migraphx::literal{migraphx::shape{f}, {eps}});
    auto var_eps  = add_common_op(mm, migraphx::make_op("add"), {variance, eps_lit});
    auto rsqrt    = mm.add_instruction(migraphx::make_op("rsqrt"), var_eps);
    auto norm     = add_common_op(mm, migraphx::make_op("mul"), {x_sub, rsqrt});
    auto scaled   = add_common_op(mm, migraphx::make_op("mul"), {norm, scale});
    add_common_op(mm, migraphx::make_op("add"), {scaled, bias});

    EXPECT(mm ==
           make_op_module("tm::layer_norm", {{"epsilon", eps}, {"axes", axes}}, mm.get_parameters()));
}
