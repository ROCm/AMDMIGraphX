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
#include <migraphx/apply_alpha_beta.hpp>

// Matrix x vector: the vector is unsqueezed to a column, dotted, then squeezed back.
static void add_mv_dot(migraphx::module& m, const std::vector<migraphx::instruction_ref>& a)
{
    auto sl1 = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), a[1]);
    auto res = migraphx::add_apply_alpha_beta(m, {a[0], sl1}, migraphx::make_op("dot"), 1.0f, 0.0f);
    auto ret = m.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), res);
    m.add_return({ret});
}

TEST_CASE(matmul_dyn_mv_test)
{
    EXPECT(check_parse("matmul_dyn_mv_test.onnx",
                       {{"1", {migraphx::shape::float_type, {{4, 8, {6}}, {7, 7}}}},
                        {"2", {migraphx::shape::float_type, {7}}}},
                       add_mv_dot));
}

TEST_CASE(matmul_sym_mv_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;
    EXPECT(check_parse("matmul_dyn_mv_test.onnx",
                       {{"1", {migraphx::shape::float_type, sym_dims({var("m", {4, 8}), lit(7)})}},
                        {"2", {migraphx::shape::float_type, {7}}}},
                       add_mv_dot));
}
