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

#include <migraphx/op/builder/quantize_dequantize_linear.hpp>

namespace {
struct test_ctx
{
    test_ctx(const std::vector<size_t>& x_lens,
             const std::vector<size_t>& s_lens,
             migraphx::shape::type_t x_type = migraphx::shape::int8_type,
             migraphx::shape::type_t s_type = migraphx::shape::float_type)
        : x_shape{x_type, x_lens},
          s_shape{s_type, s_lens},
          zp_shape{x_type, s_lens},
          axis{1},
          block_size{0}
    {
        x  = m.add_parameter("x", x_shape);
        s  = m.add_parameter("s", s_shape);
        zp = m.add_parameter("zp", zp_shape);
    }

    migraphx::module make_op_bldr()
    {
        return make_op_module(
            "dequantizelinear", {{"axis", axis}, {"block_size", block_size}}, m.get_parameters());
    }

    void expect() { EXPECT(m == make_op_bldr()); }

    migraphx::module m;
    migraphx::shape x_shape;
    migraphx::shape s_shape;
    migraphx::shape zp_shape;
    int axis;
    int block_size;

    migraphx::instruction_ref x;
    migraphx::instruction_ref s;
    migraphx::instruction_ref zp;
};

test_ctx per_tensor_ctx(const std::vector<size_t>& x_lens) { return test_ctx{x_lens, {1}}; }

test_ctx per_axis_ctx(const std::vector<size_t>& x_lens, size_t s_dim, int axis)
{
    test_ctx ctx{x_lens, {s_dim}};
    ctx.axis = axis;
    return ctx;
}

test_ctx per_axis_ctx_valid(const std::vector<size_t>& x_lens, int axis)
{
    return per_axis_ctx(x_lens, x_lens[axis], axis);
}

test_ctx blocked_ctx(const std::vector<size_t>& x_lens,
                     const std::vector<size_t>& s_lens,
                     int axis,
                     int block_size)
{
    test_ctx ctx{x_lens, s_lens};
    ctx.axis       = axis;
    ctx.block_size = block_size;
    return ctx;
}
} // namespace

// per-tensor
TEST_CASE(dequantizelinear_per_tensor_op_builder_test)
{
    auto ctx            = per_tensor_ctx({4, 3});
    migraphx::module& m = ctx.m;

    auto new_s = m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", ctx.x_shape.lens()}}), ctx.s);
    auto new_zp = m.add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", ctx.x_shape.lens()}}), ctx.zp);
    m.add_instruction(migraphx::make_op("dequantizelinear"), ctx.x, new_s, new_zp);

    ctx.expect();
}

// per-axis
TEST_CASE(dequantizelinear_per_axis_op_builder_test)
{
    auto ctx            = per_axis_ctx_valid({4, 3}, 1);
    migraphx::module& m = ctx.m;

    auto new_s = m.add_instruction(
        migraphx::make_op("broadcast", {{"axis", ctx.axis}, {"out_lens", ctx.x_shape.lens()}}),
        ctx.s);
    auto new_zp = m.add_instruction(
        migraphx::make_op("broadcast", {{"axis", ctx.axis}, {"out_lens", ctx.x_shape.lens()}}),
        ctx.zp);

    m.add_instruction(migraphx::make_op("dequantizelinear"), ctx.x, new_s, new_zp);

    ctx.expect();
}

TEST_CASE(dequantizelinear_per_axis_invalid_shapes_op_builder_test)
{
    auto ctx = per_axis_ctx({4, 3}, 5, 1);
    EXPECT(test::throws<migraphx::exception>(
        [&] { ctx.make_op_bldr(); },
        "dequantizelinear: For per axis granularity the length of y_scale (actual: 5) must be "
        "equal to size of x on axis 1(actual: 3)"));
}

TEST_CASE(dequantizelinear_per_axis_invalid_axis_op_builder_test)
{
    auto ctx = per_axis_ctx({4, 3}, 3, 8);
    EXPECT(test::throws<migraphx::exception>([&] { ctx.make_op_bldr(); },
                                             "DEQUANTIZELINEAR: axis is out of range."));
}

// blocked
TEST_CASE(dequantizelinear_blocked_op_builder_test)
{
    auto ctx            = blocked_ctx({4, 6}, {4, 3}, 1, 2);
    migraphx::module& m = ctx.m;

    auto i1 = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), ctx.s);
    i1      = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 3, 2}}}), i1);
    auto new_s = m.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 6}}}), i1);

    i1 = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), ctx.zp);
    i1 = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 3, 2}}}), i1);
    auto new_zp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 6}}}), i1);

    m.add_instruction(migraphx::make_op("dequantizelinear"), ctx.x, new_s, new_zp);

    ctx.expect();
}

TEST_CASE(dequantizelinear_blocked_invalid_axis_op_builder_test)
{
    auto ctx = blocked_ctx({4, 3}, {4, 3}, 8, 2);
    EXPECT(test::throws<migraphx::exception>([&] { ctx.make_op_bldr(); },
                                             "DEQUANTIZELINEAR: axis is out of range."));
}

TEST_CASE(dequantizelinear_blocked_invalid_rank_op_builder_test)
{
    auto ctx = blocked_ctx({3, 4, 6}, {4, 3}, 1, 2);
    EXPECT(test::throws<migraphx::exception>([&] { ctx.make_op_bldr(); },
                                             "dequantizelinear: x(rank: 3) and y_scale(rank: 2) "
                                             "must be of same rank for block granularity"));
}

TEST_CASE(dequantizelinear_blocked_invalid_shape_op_builder_test)
{
    auto ctx = blocked_ctx({4, 6}, {5, 3}, 1, 2);
    EXPECT(
        test::throws<migraphx::exception>([&] { ctx.make_op_bldr(); },
                                          "dequantizelinear: x(shape: 4, 6) and y_scale(shape: 5, "
                                          "3) shapes may only differ along provided axis(1)"));
}

TEST_CASE(dequantizelinear_blocked_invalid_blocksize_op_builder_test)
{
    auto ctx = blocked_ctx({4, 6}, {4, 3}, 1, 3);
    EXPECT(test::throws<migraphx::exception>(
        [&] { ctx.make_op_bldr(); },
        "dequantizelinear: Block size(actual: 3) must be within range [2, 2]"));
}

TEST_CASE(dequantizelinear_blocked_blocksize_zero_op_builder_test)
{
    auto ctx            = blocked_ctx({4, 6}, {4, 3}, 1, 0);
    migraphx::module& m = ctx.m;

    auto i1 = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), ctx.s);
    i1      = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 3, 2}}}), i1);
    auto new_s = m.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 6}}}), i1);

    i1 = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), ctx.zp);
    i1 = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 3, 2}}}), i1);
    auto new_zp = m.add_instruction(migraphx::make_op("reshape", {{"dims", {4, 6}}}), i1);

    m.add_instruction(migraphx::make_op("dequantizelinear"), ctx.x, new_s, new_zp);

    ctx.expect();
}

TEST_CASE(dequantizelinear_blocked_blocksize_one_op_builder_test)
{
    auto ctx            = blocked_ctx({4, 3}, {4, 3}, 1, 1);
    migraphx::module& m = ctx.m;

    m.add_instruction(migraphx::make_op("dequantizelinear"), ctx.x, ctx.s, ctx.zp);

    ctx.expect();
}
