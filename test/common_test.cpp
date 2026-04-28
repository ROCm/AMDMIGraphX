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
#include "test.hpp"
#include <migraphx/common.hpp>
#include <migraphx/module.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>

TEST_CASE(add_common_op_scalar_literal_preserves_tensor_type)
{
    migraphx::module mm;

    auto tensor =
        mm.add_parameter("x", migraphx::shape{migraphx::shape::half_type, {1, 100, 128}});
    auto scalar = mm.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {1e-5f}});

    auto result = migraphx::add_common_op(mm, migraphx::make_op("add"), {tensor, scalar});

    EXPECT(result->get_shape().type() == migraphx::shape::half_type);
}

TEST_CASE(add_common_op_scalar_literal_preserves_tensor_type_reversed)
{
    migraphx::module mm;

    auto tensor =
        mm.add_parameter("x", migraphx::shape{migraphx::shape::half_type, {1, 100, 128}});
    auto scalar = mm.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {1e-5f}});

    auto result = migraphx::add_common_op(mm, migraphx::make_op("add"), {scalar, tensor});

    EXPECT(result->get_shape().type() == migraphx::shape::half_type);
}

TEST_CASE(add_common_op_two_tensors_promotes_to_wider_type)
{
    migraphx::module mm;

    auto half_tensor =
        mm.add_parameter("x", migraphx::shape{migraphx::shape::half_type, {1, 128}});
    auto float_tensor =
        mm.add_parameter("y", migraphx::shape{migraphx::shape::float_type, {1, 128}});

    auto result =
        migraphx::add_common_op(mm, migraphx::make_op("add"), {half_tensor, float_tensor});

    EXPECT(result->get_shape().type() == migraphx::shape::float_type);
}

TEST_CASE(add_common_op_float_tensor_with_float_scalar_keeps_float)
{
    migraphx::module mm;

    auto tensor =
        mm.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {1, 100, 128}});
    auto scalar = mm.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {1e-5f}});

    auto result = migraphx::add_common_op(mm, migraphx::make_op("add"), {tensor, scalar});

    EXPECT(result->get_shape().type() == migraphx::shape::float_type);
}
