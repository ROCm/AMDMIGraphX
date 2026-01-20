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

// here only the per-tensor case is tested. The per-axis and blocked cases are tested in
// quantizelinear_test.cpp

TEST_CASE(quantizelinear_no_fp4x2_no_conversion_no_output_type_op_builder_test)
{
    migraphx::module m;

    auto x  = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3}});
    auto s  = m.add_parameter("s", migraphx::shape{migraphx::shape::float_type, {1}});
    auto zp = m.add_parameter("zp", migraphx::shape{migraphx::shape::int8_type, {1}});

    auto new_s = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 3}}}), s);
    auto new_zp =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 3}}}), zp);

    m.add_instruction(migraphx::make_op("quantizelinear"), x, new_s, new_zp);

    EXPECT(m == make_op_module("quantizelinear", {}, m.get_parameters()));
}

TEST_CASE(quantizelinear_no_fp4x2_with_conversion_no_output_type_op_builder_test)
{
    migraphx::module m;

    auto x = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3}});
    auto s = m.add_parameter(
        "s",
        migraphx::shape{migraphx::shape::half_type, {1}}); // scale_type != input_type => conversion
    auto zp = m.add_parameter("zp", migraphx::shape{migraphx::shape::int8_type, {1}});

    auto new_s = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 3}}}), s);
    auto new_zp =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 3}}}), zp);

    auto converted_scale = m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), new_s);

    m.add_instruction(migraphx::make_op("quantizelinear"), x, converted_scale, new_zp);

    EXPECT(m == make_op_module("quantizelinear", {}, m.get_parameters()));
}

TEST_CASE(quantizelinear_no_fp4x2_no_conversion_with_output_type_op_builder_test)
{
    migraphx::module m;

    auto x  = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3}});
    auto s  = m.add_parameter("s", migraphx::shape{migraphx::shape::float_type, {1}});
    auto zp = m.add_parameter("zp", migraphx::shape{migraphx::shape::int8_type, {1}});

    auto new_s = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 3}}}), s);
    auto new_zp =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 3}}}), zp);

    m.add_instruction(
        migraphx::make_op("quantizelinear", {{"out_type", migraphx::shape::int16_type}}),
        x,
        new_s,
        new_zp);

    EXPECT(m == make_op_module("quantizelinear",
                               {{"output_type", migraphx::shape::int16_type}},
                               m.get_parameters()));
}

TEST_CASE(quantizelinear_no_fp4x2_with_conversion_with_output_type_op_builder_test)
{
    migraphx::module m;

    auto x = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3}});
    auto s = m.add_parameter(
        "s",
        migraphx::shape{migraphx::shape::half_type, {1}}); // scale_type != input_type => conversion
    auto zp = m.add_parameter("zp", migraphx::shape{migraphx::shape::int8_type, {1}});

    auto new_s = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 3}}}), s);
    auto new_zp =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 3}}}), zp);

    auto converted_scale = m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), new_s);

    m.add_instruction(
        migraphx::make_op("quantizelinear", {{"out_type", migraphx::shape::int16_type}}),
        x,
        converted_scale,
        new_zp);

    EXPECT(m == make_op_module("quantizelinear",
                               {{"output_type", migraphx::shape::int16_type}},
                               m.get_parameters()));
}

TEST_CASE(quantizelinear_with_fp4x2_even_fast_axis_op_builder_test)
{
    migraphx::module m;

    auto x  = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {3, 4}});
    auto s  = m.add_parameter("s", migraphx::shape{migraphx::shape::float_type, {1}});
    auto zp = m.add_parameter("zp", migraphx::shape{migraphx::shape::int8_type, {1}});

    auto new_s = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 4}}}), s);
    auto new_zp =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {3, 4}}}), zp);

    auto q_ins = m.add_instruction(
        migraphx::make_op("quantizelinear", {{"out_type", migraphx::shape::float_type}}),
        x,
        new_s,
        new_zp);
    auto pack_ins = m.add_instruction(migraphx::make_op("pack_fp4"), q_ins);
    m.add_instruction(migraphx::make_op("unpack_fp4"), pack_ins);

    EXPECT(m == make_op_module("quantizelinear",
                               {{"output_type", migraphx::shape::fp4x2_type}},
                               m.get_parameters()));
}

TEST_CASE(quantizelinear_with_fp4x2_odd_fast_axis_op_builder_test)
{
    migraphx::module m;

    auto x  = m.add_parameter("x", migraphx::shape{migraphx::shape::float_type, {4, 3}});
    auto s  = m.add_parameter("s", migraphx::shape{migraphx::shape::float_type, {1}});
    auto zp = m.add_parameter("zp", migraphx::shape{migraphx::shape::int8_type, {1}});

    auto new_s = m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 3}}}), s);
    auto new_zp =
        m.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 3}}}), zp);

    auto q_ins = m.add_instruction(
        migraphx::make_op("quantizelinear", {{"out_type", migraphx::shape::float_type}}),
        x,
        new_s,
        new_zp);
    q_ins           = m.add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 0, 1}}}), q_ins);
    auto pack_ins   = m.add_instruction(migraphx::make_op("pack_fp4"), q_ins);
    auto unpack_ins = m.add_instruction(migraphx::make_op("unpack_fp4"), pack_ins);
    m.add_instruction(migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {3}}}),
                      unpack_ins);

    EXPECT(m == make_op_module("quantizelinear",
                               {{"output_type", migraphx::shape::fp4x2_type}},
                               m.get_parameters()));
}

namespace {
template <typename x_typ = float, typename s_typ = float, typename zp_typ = int8_t>
auto run_with_data(std::vector<size_t> x_lens,
                   std::vector<size_t> s_lens,

                   std::vector<x_typ> x_data,
                   std::vector<s_typ> s_data,
                   std::vector<zp_typ> zp_data,

                   std::optional<migraphx::shape::type_t> out_typ = std::nullopt,
                   int axis                                       = 1,
                   int block_size                                 = 0) -> std::vector<zp_typ>
{
    migraphx::module m;

    const migraphx::shape x_shape{migraphx::shape::get_type<x_typ>::value, x_lens};
    const migraphx::shape s_shape{migraphx::shape::get_type<s_typ>::value, s_lens};
    const migraphx::shape zp_shape{migraphx::shape::get_type<zp_typ>::value, s_lens};

    m.add_parameter("x", x_shape);
    m.add_parameter("s", s_shape);
    m.add_parameter("zp", zp_shape);

    if(out_typ.has_value())
    {
        m = make_op_module(
            "quantizelinear",
            {{"axis", axis}, {"block_size", block_size}, {"output_type", out_typ.value()}},
            m.get_parameters());
    }
    else
    {
        m = make_op_module(
            "quantizelinear", {{"axis", axis}, {"block_size", block_size}}, m.get_parameters());
    }

    migraphx::program p{std::move(m)};
    p.compile(migraphx::make_target("ref"));

    migraphx::parameter_map params;
    params["x"]  = migraphx::argument(x_shape, x_data.data());
    params["s"]  = migraphx::argument(s_shape, s_data.data());
    params["zp"] = migraphx::argument(zp_shape, zp_data.data());

    auto result = p.eval(params).back();
    std::vector<zp_typ> result_data(result.get_shape().elements());
    result.visit(
        [&](auto output) { std::copy(output.begin(), output.end(), result_data.begin()); });

    return result_data;
}
} // namespace

// verify tests
TEST_CASE(quantizelinear_verify_per_tensor_op_builder_test)
{
    /*
        y = round(x / s) + zp
        more details on how the calculation is done can be found in dequantizelinear_test.cpp
    */

    std::vector<float> x = {
        -1.0f, -0.7f, -0.5f, -0.2f, 0.0f, 0.4f, 0.5f, 0.75f, 1.0f, 1.2f, 3.5f, -1.5f};
    std::vector<float> s   = {0.0078125f};
    std::vector<int8_t> zp = {0};

    std::vector<int8_t> expected = {-128, -90, -64, -26, 0, 51, 64, 96, 127, 127, 127, -128};

    auto result = run_with_data({4, 3}, {1}, x, s, zp);

    EXPECT(migraphx::verify::verify_rms_range(result, expected));
}

TEST_CASE(quantizelinear_verify_per_tensor_fp4x2_op_builder_test)
{
    std::vector<float> x = {
        -1000.0f, -2.399f, -2.25f, -1.5f, -1.2f, -0.8f, -0.5f, -0.3f, 0.0f, 0.3f, 1.2f, 1234.567f};
    std::vector<float> s    = {0.3f};
    std::vector<int64_t> zp = {2};

    std::vector<int8_t> expected = {-6, -6, -4, -3, -2, -1, 0, 1, 2, 3, 6, 6};

    auto result = run_with_data({4, 3}, {1}, x, s, zp, migraphx::shape::fp4x2_type);

    EXPECT(migraphx::verify::verify_rms_range(result, expected));
}

TEST_CASE(quantizelinear_verify_per_axis_op_builder_test)
{
    std::vector<float> x = {
        -1.0f, -0.7f, -0.5f, -0.2f, 0.0f, 0.4f, 0.5f, 0.75f, 1.0f, 1.2f, 3.5f, -1.5f};
    std::vector<float> s   = {0.1f, 1.0f, 10.0f};
    std::vector<int8_t> zp = {2, 0, 30};

    std::vector<int8_t> expected = {-8, -1, 30, 0, 0, 30, 7, 1, 30, 14, 4, 30};

    auto result = run_with_data({4, 3}, {3}, x, s, zp);

    EXPECT(migraphx::verify::verify_rms_range(result, expected));
}

TEST_CASE(quantizelinear_verify_blocked_op_builder_test)
{
    std::vector<float> x = {
        -1.0f, -0.7f, -0.5f, -0.2f, 0.0f, 0.4f, 0.5f, 0.75f, 1.0f, 1.2f, 3.5f, -1.5f,
        -1.0f, -0.7f, -0.5f, -0.2f, 0.0f, 0.4f, 0.5f, 0.75f, 1.0f, 1.2f, 3.5f, -1.5f,
    };
    std::vector<float> s = {
        1.0f, 0.1f, 10.0f, 2.0f, 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 10.0f, 0.1f};
    std::vector<int8_t> zp = {1, 10, 0, 10, 0, 1, 0, 0, 0, 0, 1, 0};

    std::vector<int8_t> expected = {0,    0,    5,    8,    0,   0,   10,  10,  10, 12, 127, -128,
                                    -128, -128, -128, -128, 127, 127, 127, 127, 1,  1,  35,  -15};

    auto result = run_with_data({4, 6}, {4, 3}, x, s, zp, std::nullopt, 1, 2);

    EXPECT(migraphx::verify::verify_rms_range(result, expected));
}
