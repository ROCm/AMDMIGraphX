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
#include <migraphx/literal.hpp>

// gather_elements flattens the data and gathers element-wise using per-element flat
// offsets: shape_index + (index - axis_coord) * axis_stride, evaluated over the index shape.
TEST_CASE(gather_elements_op_builder_test)
{
    const auto f = migraphx::shape::float_type;
    const auto i = migraphx::shape::int32_type;

    migraphx::module mm;
    auto data = mm.add_parameter("data", {f, {2, 3}});
    auto ind  = mm.add_parameter("ind", {i, {2, 3}});

    auto arg_data = mm.add_instruction(migraphx::make_op("contiguous"), data);
    auto arg_ind  = mm.add_instruction(migraphx::make_op("contiguous"), ind);
    arg_data      = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {6}}}), arg_data);

    std::vector<int64_t> shape_idx = {0, 1, 2, 3, 4, 5};
    std::vector<int64_t> dim_idx   = {0, 1, 2, 0, 1, 2};
    auto l_shape_idx = mm.add_literal(migraphx::literal{migraphx::shape{i, {2, 3}}, shape_idx});
    auto l_dim_idx   = mm.add_literal(migraphx::literal{migraphx::shape{i, {2, 3}}, dim_idx});
    auto l_stride    = mm.add_literal(migraphx::literal{migraphx::shape{i, {1}}, {1}});
    l_stride =
        mm.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3}}}), l_stride);
    auto dim_diff = mm.add_instruction(migraphx::make_op("sub"), arg_ind, l_dim_idx);
    auto delta    = mm.add_instruction(migraphx::make_op("mul"), dim_diff, l_stride);
    auto indices  = mm.add_instruction(migraphx::make_op("add"), l_shape_idx, delta);
    mm.add_instruction(migraphx::make_op("gather", {{"axis", 0}}), arg_data, indices);

    EXPECT(mm == make_op_module("gather_elements", {{"axis", 1}}, mm.get_parameters()));
}
