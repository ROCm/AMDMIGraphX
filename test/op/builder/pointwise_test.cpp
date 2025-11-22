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

namespace {
const std::vector<std::string>& pointwise_op_names()
{
    static const std::vector<std::string> op_names_set{"add",
                                                       "div",
                                                       "logical_and",
                                                       "logical_or",
                                                       "logical_xor",
                                                       "bitwise_and",
                                                       "mul",
                                                       "prelu",
                                                       "sub"};
    return op_names_set;
}
} // namespace

TEST_CASE(pointwise_not_broadcasted_op_builder_test)
{
    std::for_each(
        pointwise_op_names().begin(), pointwise_op_names().end(), [&](const std::string& op_name) {
            migraphx::module mm;

            auto a_arg = mm.add_parameter("a", {migraphx::shape::int64_type, {2, 4}});
            auto b_arg = mm.add_parameter("b", {migraphx::shape::int64_type, {2, 4}});

            add_common_op(mm, migraphx::make_op(op_name), {a_arg, b_arg});

            EXPECT(mm == make_op_module(op_name, {}, mm.get_parameters()));
        });
}

TEST_CASE(pointwise_not_broadcasted_implicit_broadcast_op_builder_test)
{
    std::for_each(
        pointwise_op_names().begin(), pointwise_op_names().end(), [&](const std::string& op_name) {
            migraphx::module mm;

            auto a_arg = mm.add_parameter("a", {migraphx::shape::int64_type, {2, 4}});
            auto b_arg = mm.add_parameter("b", {migraphx::shape::int64_type, {2, 1}});

            add_common_op(mm, migraphx::make_op(op_name), {a_arg, b_arg});

            EXPECT(mm == make_op_module(op_name, {}, mm.get_parameters()));
        });
}

TEST_CASE(pointwise_non_zero_broadcasted_op_builder_test)
{
    std::for_each(
        pointwise_op_names().begin(), pointwise_op_names().end(), [&](const std::string& op_name) {
            migraphx::module mm;

            auto a_arg = mm.add_parameter("a", {migraphx::shape::int64_type, {2, 3, 4, 5}});
            auto b_arg = mm.add_parameter("b", {migraphx::shape::int64_type, {3, 4}});

            auto l = mm.add_instruction(
                migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {2, 3, 4, 5}}}), b_arg);
            mm.add_instruction(migraphx::make_op(op_name), {a_arg, l});

            EXPECT(mm == make_op_module(op_name, {{"broadcasted_axis", 1}}, mm.get_parameters()));
        });
}
