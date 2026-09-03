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

TEST_CASE(matmul_dyn_broadcast_test)
{
    EXPECT(check_parse(
        "matmul_dyn_broadcast_test.onnx",
        {{"1", {migraphx::shape::float_type, {7}}},
         {"2", {migraphx::shape::float_type, {{5, 5}, {7, 7}, {4, 8, {6}}}}}},
        [](migraphx::module& m, const auto& a) {
            auto u  = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), a[0]);
            auto b0 = m.add_instruction(migraphx::make_op("broadcast_for_dot"), u, a[1]);
            auto b1 = m.add_instruction(migraphx::make_op("broadcast_for_dot"), a[1], u);
            auto d  = m.add_instruction(migraphx::make_op("dot"), b0, b1);
            m.add_return({m.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), d)});
        }));
}

TEST_CASE(matmul_sym_broadcast_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;
    EXPECT(check_parse(
        "matmul_dyn_broadcast_test.onnx",
        {{"1", {migraphx::shape::float_type, {7}}},
         {"2", {migraphx::shape::float_type, sym_dims({lit(5), lit(7), var("k", {4, 8})})}}},
        [](migraphx::module& m, const auto& a) {
            // Same shape-donor form as the range-dynamic case, so the broadcast resolves
            // from its inputs once they are static
            auto u  = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), a[0]);
            auto b0 = m.add_instruction(migraphx::make_op("broadcast_for_dot"), u, a[1]);
            auto b1 = m.add_instruction(migraphx::make_op("broadcast_for_dot"), a[1], u);
            auto d  = m.add_instruction(migraphx::make_op("dot"), b0, b1);
            m.add_return({m.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), d)});
        }));
}
