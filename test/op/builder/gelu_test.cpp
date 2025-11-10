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
#include <migraphx/instruction.hpp>

TEST_CASE(gelu_quick_happy_path_op_builder_test)
{
    migraphx::module mm;
    const float alpha_val = 0.5f;

    auto x     = mm.add_parameter("x", {migraphx::shape::float_type, {3, 3}});
    auto alpha = mm.add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {alpha_val}});
    auto mul_alpha = add_common_op(mm, migraphx::make_op("mul"), {alpha, x});
    auto sigmoid   = mm.add_instruction(migraphx::make_op("sigmoid"), {mul_alpha});
    add_common_op(mm, migraphx::make_op("mul"), {x, sigmoid});

    EXPECT(mm == make_op_module("gelu_quick", {{"alpha", alpha_val}}, mm.get_parameters()));
}

TEST_CASE(gelu_erf_happy_path_op_builder_test)
{
    migraphx::module mm;

    auto x        = mm.add_parameter("x", {migraphx::shape::float_type, {3, 3}});
    auto half     = mm.add_literal({migraphx::shape::float_type, {0.5f}});
    auto one      = mm.add_literal({migraphx::shape::float_type, {1.0f}});
    auto sqrt2    = mm.add_literal({migraphx::shape::float_type, {static_cast<float>(M_SQRT2)}});
    auto mul_half = add_common_op(mm, migraphx::make_op("mul"), {x, half});
    auto div      = add_common_op(mm, migraphx::make_op("div"), {x, sqrt2});
    auto erf      = mm.add_instruction(migraphx::make_op("erf"), div);
    auto add_one  = add_common_op(mm, migraphx::make_op("add"), {erf, one});
    add_common_op(mm, migraphx::make_op("mul"), {mul_half, add_one});

    EXPECT(mm == make_op_module("gelu_erf", mm.get_parameters()));
}

TEST_CASE(gelu_tanh_fast_happy_path_op_builder_test)
{
    migraphx::module mm;

    auto x = mm.add_parameter("x", {migraphx::shape::float_type, {3, 3}});

    const auto fit_const_val  = 0.035677;
    auto fit_const            = mm.add_literal({migraphx::shape::float_type, {fit_const_val}});
    const auto sqrt_2_rpi_val = 0.797885;
    auto sqrt_2_rpi           = mm.add_literal({migraphx::shape::float_type, {sqrt_2_rpi_val}});
    auto one                  = mm.add_literal({migraphx::shape::float_type, {1.0f}});
    auto half                 = mm.add_literal({migraphx::shape::float_type, {0.5f}});
    auto three                = mm.add_literal({migraphx::shape::float_type, {3.0f}});

    // [0.044715|0.035677] * x^3
    auto pow0 = add_common_op(mm, migraphx::make_op("pow"), {x, three});
    auto mul0 = add_common_op(mm, migraphx::make_op("mul"), {pow0, fit_const});
    migraphx::instruction_ref tanh_in;

    // approx = 0.797885 * x + 0.035677 * x^3
    auto mul1 = add_common_op(mm, migraphx::make_op("mul"), {sqrt_2_rpi, x});
    tanh_in   = add_common_op(mm, migraphx::make_op("add"), {mul0, mul1});

    // 0.5 * x * (1 + Tanh(approx))
    auto tanh0 = add_common_op(mm, migraphx::make_op("tanh"), {tanh_in});
    auto add1  = add_common_op(mm, migraphx::make_op("add"), {tanh0, one});
    auto mul2  = add_common_op(mm, migraphx::make_op("mul"), {x, half});
    add_common_op(mm, migraphx::make_op("mul"), {add1, mul2});

    EXPECT(mm == make_op_module("gelu_tanh", {{"fast", true}}, mm.get_parameters()));
}

TEST_CASE(gelu_tanh_not_fast_happy_path_op_builder_test)
{
    migraphx::module mm;

    auto x = mm.add_parameter("x", {migraphx::shape::float_type, {3, 3}});

    const auto fit_const_val  = 0.044715;
    auto fit_const            = mm.add_literal({migraphx::shape::float_type, {fit_const_val}});
    const auto sqrt_2_rpi_val = sqrt(M_2_PI);
    auto sqrt_2_rpi           = mm.add_literal({migraphx::shape::float_type, {sqrt_2_rpi_val}});
    auto one                  = mm.add_literal({migraphx::shape::float_type, {1.0f}});
    auto half                 = mm.add_literal({migraphx::shape::float_type, {0.5f}});
    auto three                = mm.add_literal({migraphx::shape::float_type, {3.0f}});

    // [0.044715|0.035677] * x^3
    auto pow0 = add_common_op(mm, migraphx::make_op("pow"), {x, three});
    auto mul0 = add_common_op(mm, migraphx::make_op("mul"), {pow0, fit_const});
    migraphx::instruction_ref tanh_in;

    // approx = sqrt(2/pi) * (x + 0.044715 * x^3
    auto add0 = add_common_op(mm, migraphx::make_op("add"), {mul0, x});
    tanh_in   = add_common_op(mm, migraphx::make_op("mul"), {add0, sqrt_2_rpi});

    // 0.5 * x * (1 + Tanh(approx))
    auto tanh0 = add_common_op(mm, migraphx::make_op("tanh"), {tanh_in});
    auto add1  = add_common_op(mm, migraphx::make_op("add"), {tanh0, one});
    auto mul2  = add_common_op(mm, migraphx::make_op("mul"), {x, half});
    add_common_op(mm, migraphx::make_op("mul"), {add1, mul2});

    EXPECT(mm == make_op_module("gelu_tanh", {{"fast", false}}, mm.get_parameters()));
}

TEST_CASE(gelu_split_happy_path_op_builder_path)
{
    migraphx::module mm;

    auto x                     = mm.add_parameter("x", {migraphx::shape::float_type, {2, 4, 6}});
    const size_t last_dim_size = x->get_shape().lens().back();
    auto split_left            = mm.add_instruction(
        migraphx::make_op("slice",
                                     {{"axes", {-1}}, {"starts", {0}}, {"ends", {last_dim_size / 2}}}),
        x);
    auto split_right = mm.add_instruction(
        migraphx::make_op(
            "slice", {{"axes", {-1}}, {"starts", {last_dim_size / 2}}, {"ends", {last_dim_size}}}),
        x);

    // building up gelu_erf
    migraphx::instruction_ref gelu_erf;
    {
        auto x2    = split_right;
        auto half  = mm.add_literal({migraphx::shape::float_type, {0.5f}});
        auto one   = mm.add_literal({migraphx::shape::float_type, {1.0f}});
        auto sqrt2 = mm.add_literal({migraphx::shape::float_type, {static_cast<float>(M_SQRT2)}});
        auto mul_half = add_common_op(mm, migraphx::make_op("mul"), {x2, half});
        auto div      = add_common_op(mm, migraphx::make_op("div"), {x2, sqrt2});
        auto erf      = mm.add_instruction(migraphx::make_op("erf"), div);
        auto add_one  = add_common_op(mm, migraphx::make_op("add"), {erf, one});
        gelu_erf      = add_common_op(mm, migraphx::make_op("mul"), {mul_half, add_one});
    }

    add_common_op(mm, migraphx::make_op("mul"), {split_left, gelu_erf});

    EXPECT(mm == make_op_module("gelu_split", mm.get_parameters()));
}

TEST_CASE(gelu_split_invalid_dimension_op_builder_path)
{
    migraphx::module mm;
    mm.add_parameter("x", {migraphx::shape::float_type, {3, 3}});
    EXPECT(test::throws<migraphx::exception>(
        [&] { make_op_module("gelu_split", mm.get_parameters()); },
        "gelu_split op_builder: BiasSplitGelu must have even last dimension which is >= 2"));
}
