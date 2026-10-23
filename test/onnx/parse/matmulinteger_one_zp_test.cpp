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

#include <onnx_test.hpp>

TEST_CASE(matmulinteger_one_zp_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto l2 = mm->add_literal(
        migraphx::literal(migraphx::shape{migraphx::shape::int8_type, {1}, {0}}, {5}));

    auto l0 = mm->add_parameter("1", migraphx::shape{migraphx::shape::int8_type, {4, 3}});
    auto l1 = mm->add_parameter("2", migraphx::shape{migraphx::shape::uint8_type, {3, 2}});

    auto za = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1}}}), l2);
    za      = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), za);
    auto zero =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {0}});
    auto offset =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {128}});
    auto zb     = migraphx::add_common_op(*mm, migraphx::make_op("sub"), {zero, offset});
    auto wide_b = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), l1);
    auto input_offset =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {128}});
    auto shifted_b = migraphx::add_common_op(*mm, migraphx::make_op("sub"), {wide_b, input_offset});
    auto qb        = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}), shifted_b);
    auto result = mm->add_instruction(migraphx::make_op("quant_dot"), l0, qb);
    auto wb     = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), qb);
    auto sum_b        = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), wb);
    auto correction_a = migraphx::add_common_op(*mm, migraphx::make_op("mul"), {za, sum_b});
    result  = migraphx::add_common_op(*mm, migraphx::make_op("sub"), {result, correction_a});
    auto wa = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int32_type}}), l0);
    auto sum_a        = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), wa);
    auto correction_b = migraphx::add_common_op(*mm, migraphx::make_op("mul"), {sum_a, zb});
    result = migraphx::add_common_op(*mm, migraphx::make_op("sub"), {result, correction_b});
    auto k = mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int32_type}, {3}});
    auto correction = migraphx::add_common_op(*mm, migraphx::make_op("mul"), {za, zb});
    correction      = migraphx::add_common_op(*mm, migraphx::make_op("mul"), {correction, k});
    migraphx::add_common_op(*mm, migraphx::make_op("add"), {result, correction});

    auto prog = optimize_onnx("matmulinteger_int8_uint8_one_zp_test.onnx");

    EXPECT(p.sort() == prog.sort());
}
