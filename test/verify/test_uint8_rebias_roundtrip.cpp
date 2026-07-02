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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>

// Isolates the uint8 -> int8 rebias chain simplify_qdq emits for asymmetric activations:
// quantizelinear(uint8) -> convert(int32) -> sub(128) -> convert(int8), with no conv or MLIR.
struct test_uint8_rebias_roundtrip : verify_program<test_uint8_rebias_roundtrip>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        migraphx::shape x_shape{migraphx::shape::float_type, {2, 8}};
        auto x = mm->add_parameter("x", x_shape);

        auto scale = mm->add_literal(migraphx::literal{migraphx::shape::float_type, {0.05f}});
        auto zp    = mm->add_literal(migraphx::literal{migraphx::shape::uint8_type, {0}});
        auto scale_b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x_shape.lens()}}), scale);
        auto zp_b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x_shape.lens()}}), zp);

        auto q = mm->add_instruction(migraphx::make_op("quantizelinear"), x, scale_b, zp_b);

        // Rebias to int8 by subtracting 128 through int32 (same IR as rebias_uint8_to_int8).
        auto q_i32 = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), q);
        auto k128 = mm->add_literal(migraphx::literal{migraphx::shape::int32_type, {128}});
        auto k128_b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x_shape.lens()}}), k128);
        auto diff = mm->add_instruction(migraphx::make_op("sub"), q_i32, k128_b);
        auto q_i8 = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}), diff);

        auto out = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), q_i8);
        mm->add_return({out});
        return p;
    }
    std::string section() const { return "conv"; }
};
