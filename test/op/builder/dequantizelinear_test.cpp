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

#include <op_builder_test_utils.hpp>
#include <migraphx/op/builder/quantize_dequantize_linear.hpp>

#include <migraphx/program.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/verify.hpp>

namespace {
template <typename XT = int8_t, typename ST = float>
struct test_context
{
    using x_typ = XT;
    using s_typ = ST;

    test_context(const std::vector<size_t>& x_lens,
             const std::vector<size_t>& s_lens,
             migraphx::shape::type_t x_type = migraphx::shape::get_type<x_typ>::value,
             migraphx::shape::type_t s_type = migraphx::shape::get_type<s_typ>::value)
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

    void expect_verify(const std::vector<float>& expected, const std::vector<float>& result)
    {
        EXPECT(migraphx::verify::verify_rms_range(result, expected));
    }

    std::vector<float>
    run_with_data(std::vector<x_typ> x_data, std::vector<s_typ> s_data, std::vector<x_typ> zp_data)
    {
        m = make_op_bldr();
        migraphx::program p{std::move(m)};
        p.compile(migraphx::make_target("ref"));

        migraphx::parameter_map params;
        params["x"]  = migraphx::argument(x_shape, x_data.data());
        params["s"]  = migraphx::argument(s_shape, s_data.data());
        params["zp"] = migraphx::argument(zp_shape, zp_data.data());

        auto result = p.eval(params).back();
        std::vector<float> result_data(result.get_shape().elements());
        result.visit(
            [&](auto output) { std::copy(output.begin(), output.end(), result_data.begin()); });
        return result_data;
    }

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

template <typename XT = int8_t, typename ST = float>
test_context<XT, ST> per_tensor_ctx(const std::vector<size_t>& x_lens)
{
    return test_context<XT, ST>{x_lens, {1}};
}

template <typename XT = int8_t, typename ST = float>
test_context<XT, ST> per_axis_ctx(const std::vector<size_t>& x_lens, size_t s_dim, int axis)
{
    test_context<XT, ST> ctx{x_lens, {s_dim}};
    ctx.axis = axis;
    return ctx;
}

template <typename XT = int8_t, typename ST = float>
test_context<XT, ST> per_axis_ctx_valid(const std::vector<size_t>& x_lens, int axis)
{
    return per_axis_ctx<XT, ST>(x_lens, x_lens[axis], axis);
}

template <typename XT = int8_t, typename ST = float>
test_context<XT, ST> blocked_ctx(const std::vector<size_t>& x_lens,
                             const std::vector<size_t>& s_lens,
                             int axis,
                             int block_size)
{
    test_context<XT, ST> ctx{x_lens, s_lens};
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

// verify tests
// per-tensor
TEST_CASE(dequantizelinear_verify_per_tensor_op_builder_test)
{
    /*
    y = (x - zp) * s

    same 's' and 'zp' scalar is applied to each and every element of the input tensor

    E.g.  For x = 64, zp = -128, s = 0.1
              y = (64 - (-128)) * 0.1 = 19.2

    input:
    {
        -128,   -64,    0,
        64,     127,    0,
        64,     -64,    -128,
        32,     -32,    16
    }

    expected output:
    {
        0,      6.4,    12.8,
        19.2,   25.5,   12.8,
        19.2,   6.4,    0,
        16,     9.6,    14.4
    }
    */

    auto ctx = per_tensor_ctx({4, 3});

    std::vector<int8_t> x  = {-128, -64, 0, 64, 127, 0, 64, -64, -128, 32, -32, 16};
    std::vector<float> s   = {0.1f};
    std::vector<int8_t> zp = {-128};

    std::vector<float> expected_result = {
        0, 6.4, 12.8, 19.2, 25.5, 12.8, 19.2, 6.4, 0, 16, 9.6, 14.4};

    auto result = ctx.run_with_data(x, s, zp);
    ctx.expect_verify(expected_result, result);
}

// per-axis
TEST_CASE(dequantizelinear_verify_per_axis_op_builder_test)
{
    /*
    different scale and zero-point is applied for the elements of the tensor along the specified
    axis E.g.: s[1] = 1.0 and zp[1] = 1 will be applied for column 1, so: x = -64, zp = 1 y = (-64 -
    (1)) * 1.0 = -65

    input scale :       {0.1, 1.0, 10.0}
    input zero-points:  {-128, 1, 64 }
    'axis': 1

    input:
    {
        -128,   -64,    0,
        64,     127,    0,
        64,     -64,    -128,
        32,     -32,    16
    }

    expected output:
    {
        0,      -65,    -640,
        19.2,   126,    -640,
        19.2,   -65,    -1920,
        16,     -33,    -480
    }
    */

    auto ctx = per_axis_ctx_valid({4, 3}, 1);

    std::vector<int8_t> x  = {-128, -64, 0, 64, 127, 0, 64, -64, -128, 32, -32, 16};
    std::vector<float> s   = {0.1f, 1.0f, 10.0f};
    std::vector<int8_t> zp = {-128, 1, 64};

    std::vector<float> expected_result = {
        0, -65, -640, 19.2, 126, -640, 19.2, -65, -1920, 16, -33, -480};

    auto result = ctx.run_with_data(x, s, zp);
    ctx.expect_verify(expected_result, result);
}

// blocked
TEST_CASE(dequantizelinear_verify_blocked_op_builder_test)
{
    /*
    input:
    {
        -128,   -64,    0,      64,    127,     0,
        64,     -64, -128,      32,    -32,     16,
        -16,    -32,  -64,    -128,      0,     64,
        127,    0,     64,     -64,    -128,    32
    }

    the input will be split into blocks along the specified axis:
    input_sliced:
    {
        -128, -64,          0,  64,          127,  0,
        64,   -64,       -128,  32,          -32, 16,
        -16,  -32,        -64,-128,            0, 64,
        127,    0,         64, -64,         -128, 32
    }

    the scales and zero-points will be applied per block
    E.g. for block 0 (elements along axis 1: -128, -64),
         s[0] = 1.0, zp[0] = 1
         so for x = -64,
         y = (-64 - (1)) * 1.0 = -65

    or
    for block 1 in the last row (elements along axis 1: 64, -64),
         s[3,1] = 10, zp[3,1] = 1
         so for x = 64,
         y = (64 - (1)) * 10 = 5.4
         and for x = -64
         y = (-64 - (1)) * 10 = -650

    expected output:
    {
        -129, -65,      -1, 5.4,        1270, 0,
        108, -148,    12.8, 3.2,           0, 0,
        0, 0,            0,   0,           0, 0,
        0, 0,          630,-650,       -12.8, 3.2
    }
    */

    auto ctx = blocked_ctx(/*x_lens*/ {4, 6}, /*s_lens*/ {4, 3}, /*axis*/ 1, /*block_size*/ 2);

    std::vector<int8_t> x = {-128, -64, 0,   64,   127, 0,  64,  -64, -128, 32,  -32,  16,
                             -16,  -32, -64, -128, 0,   64, 127, 0,   64,   -64, -128, 32};
    std::vector<float> s  = {
        1.0f, 0.1f, 10.0f, 2.0f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10.0f, 0.1f};
    std::vector<int8_t> zp = {1, 10, 0, 10, 0, 1, 0, 0, 0, 0, 1, 0};

    std::vector<float> expected_result = {-129,  -65, -1, 5.4, 1270, 0,    108,   -148,
                                          -12.8, 3.2, 0,  0,   0,    0,    0,     0,
                                          0,     0,   0,  0,   630,  -650, -12.8, 3.2};

    auto result = ctx.run_with_data(x, s, zp);
    ctx.expect_verify(expected_result, result);
}
