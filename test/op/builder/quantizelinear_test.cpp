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

TEST_CASE(quantizelinear_with_fp4x2_no_odd_fast_axis_op_builder_test)
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
