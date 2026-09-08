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

// TODO: use shapes that require broadcasting, when available.
static void build_where(migraphx::module& m, const std::vector<migraphx::instruction_ref>& a)
{
    m.add_return({m.add_instruction(migraphx::make_op("where"), a[0], a[1], a[2])});
}

TEST_CASE(where_dyn_test)
{
    EXPECT(check_parse("where_dyn_test.onnx",
                       {{"c", {migraphx::shape::bool_type, {{1, 4}, {2, 2}, {2, 2}}}},
                        {"x", {migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}}}},
                        {"y", {migraphx::shape::float_type, {{1, 4}, {2, 2}, {2, 2}}}}},
                       build_where));
}

TEST_CASE(where_sym_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;
    auto d = [] { return sym_dims({var("n", {1, 4}), lit(2), lit(2)}); };
    EXPECT(check_parse("where_dyn_test.onnx",
                       {{"c", {migraphx::shape::bool_type, d()}},
                        {"x", {migraphx::shape::float_type, d()}},
                        {"y", {migraphx::shape::float_type, d()}}},
                       build_where));
}
