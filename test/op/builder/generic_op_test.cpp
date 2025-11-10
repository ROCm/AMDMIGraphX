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
const std::vector<std::string>& generic_op_names()
{
    static const std::vector<std::string> op_names_set{
        "abs", "acos", "acosh", "asin",  "asinh", "atan",      "atanh",    "ceil",    "concat",
        "cos", "cosh", "elu",   "erf",   "exp",   "floor",     "identity", "isnan",   "leaky_relu",
        "log", "lrn",  "neg",   "recip", "relu",  "nearbyint", "rsqrt",    "sigmoid", "sign",
        "sin", "sinh", "sqrt",  "tan",   "tanh",  "not"};
    return op_names_set;
}
} // namespace

TEST_CASE(generic_not_continuous_op_builder_test)
{
    std::for_each(
        generic_op_names().begin(), generic_op_names().end(), [&](const std::string& op_name) {
            migraphx::module mm;
            auto a_arg     = mm.add_parameter("a", {migraphx::shape::int64_type, {2, 4, 3, 5}});
            const auto& op = migraphx::make_op(op_name);
            mm.add_instruction(op, a_arg);

            EXPECT(mm == make_op_module(op_name, migraphx::to_value(op), mm.get_parameters()));
        });
}

TEST_CASE(generic_not_continuous_gathernd_op_builder_test)
{
    migraphx::module mm;
    auto l0 = mm.add_parameter("data", migraphx::shape{migraphx::shape::float_type, {2, 2}});
    auto l1 = mm.add_parameter("indices", migraphx::shape{migraphx::shape::int64_type, {2, 2}});
    const auto& op = migraphx::make_op("gathernd");
    mm.add_instruction(op, l0, l1);

    EXPECT(mm == make_op_module("gathernd", migraphx::to_value(op), mm.get_parameters()));
}

TEST_CASE(generic_continuous_flatten_op_builder_test)
{
    migraphx::module mm;
    auto l0 = mm.add_parameter(
        "0", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}, {5, 5}, {6, 6}}});
    auto cont_l0   = mm.add_instruction(migraphx::make_op("contiguous"), l0);
    const auto& op = migraphx::make_op("flatten");
    mm.add_instruction(op, cont_l0);

    EXPECT(mm == make_op_module("flatten", migraphx::to_value(op), mm.get_parameters()));
}

TEST_CASE(generic_continuous_gather_op_builder_test)
{
    migraphx::module mm;
    auto l0 = mm.add_parameter(
        "data", migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}, {5, 5}, {6, 6}}});
    auto l1 = mm.add_parameter(
        "indices", migraphx::shape{migraphx::shape::int32_type, {{1, 4}, {3, 3}, {4, 4}, {5, 5}}});
    auto cont_l0   = mm.add_instruction(migraphx::make_op("contiguous"), l0);
    auto cont_l1   = mm.add_instruction(migraphx::make_op("contiguous"), l1);
    const auto& op = migraphx::make_op("gather", {{"axis", 1}});
    mm.add_instruction(op, cont_l0, cont_l1);

    EXPECT(mm == make_op_module("gather", migraphx::to_value(op), mm.get_parameters()));
}
