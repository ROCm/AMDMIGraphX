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

// tm::glu splits the input in half along `axis` and gates the first half by
// sigmoid of the second: glu(x) = x1 * sigmoid(x2).
TEST_CASE(torch_kit_glu_op_builder_test)
{
    const auto f = migraphx::shape::float_type;
    migraphx::module mm;
    auto x     = mm.add_parameter("x", {f, {2, 4}});
    auto first = mm.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {2}}}), x);
    auto second = mm.add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {4}}}), x);
    auto gate = mm.add_instruction(migraphx::make_op("sigmoid"), second);
    add_common_op(mm, migraphx::make_op("mul"), {first, gate});

    EXPECT(mm == make_op_module("tm::glu", {{"axis", -1}}, mm.get_parameters()));
}
