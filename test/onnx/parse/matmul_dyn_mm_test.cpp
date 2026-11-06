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

static void add_dot(migraphx::module& m, const std::vector<migraphx::instruction_ref>& a)
{
    auto ret =
        migraphx::add_apply_alpha_beta(m, {a[0], a[1]}, migraphx::make_op("dot"), 1.0f, 0.0f);
    m.add_return({ret});
}

TEST_CASE(matmul_dyn_mm_test)
{
    EXPECT(check_parse("matmul_dyn_mm_test.onnx",
                       {{"1", {migraphx::shape::float_type, {{4, 8, {6}}, {7, 7}}}},
                        {"2", {migraphx::shape::float_type, {{7, 7}, {1, 5, {3}}}}}},
                       add_dot));
}

TEST_CASE(matmul_sym_mm_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;
    EXPECT(check_parse("matmul_dyn_mm_test.onnx",
                       {{"1", {migraphx::shape::float_type, sym_dims({var("m", {4, 8}), lit(7)})}},
                        {"2", {migraphx::shape::float_type, sym_dims({lit(7), var("n", {1, 5})})}}},
                       add_dot));
}
