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
#include <migraphx/make_op.hpp>

// ND linear flattens to rank 2, delegates to gemm, then reshapes back.
TEST_CASE(torch_kit_linear_op_builder_test)
{
    const auto f = migraphx::shape::float_type;
    migraphx::module mm;
    auto x    = mm.add_parameter("x", {f, {2, 3, 4}});
    auto w    = mm.add_parameter("w", {f, {5, 4}});
    auto bias = mm.add_parameter("bias", {f, {5}});
    auto x2d  = mm.add_instruction(migraphx::make_op("reshape", {{"dims", {6, 4}}}), x);
    auto out  = migraphx::op::builder::add("gemm", mm, {x2d, w, bias}, {{"transB", true}}).front();
    mm.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 5}}}), out);

    EXPECT(mm == make_op_module("tm::linear", mm.get_parameters()));
}

// rank-2 linear is exactly the gemm builder.
TEST_CASE(torch_kit_linear_no_bias_op_builder_test)
{
    const auto f = migraphx::shape::float_type;
    migraphx::module mm;
    auto x = mm.add_parameter("x", {f, {3, 4}});
    auto w = mm.add_parameter("w", {f, {5, 4}});
    migraphx::op::builder::add("gemm", mm, {x, w}, {{"transB", true}});

    EXPECT(mm == make_op_module("tm::linear", mm.get_parameters()));
}
