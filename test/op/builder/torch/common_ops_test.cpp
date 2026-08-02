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

#include <algorithm>
#include <iterator>
#include <string>
#include <vector>
#include <op_builder_test_utils.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/common.hpp>

// The torch kit registers the common (broadcast/convert) ops and a set of plain passthrough
// ops under the "tm::" prefix. Each builder must insert exactly its wrapped op over the args.

namespace {
struct param_spec
{
    std::string name;
    migraphx::shape shape;
};

// Verifies a plain (non-common) builder inserts exactly the wrapped op over the given args.
// Returns the comparison so the caller can EXPECT() it with the op name as a literal -- a failure
// message then identifies which op did not match.
bool check_plain_op(const std::string& op_name,
                    const migraphx::value& options,
                    const std::vector<param_spec>& params)
{
    migraphx::module mm;
    std::vector<migraphx::instruction_ref> args;
    std::transform(params.begin(), params.end(), std::back_inserter(args), [&](const auto& p) {
        return mm.add_parameter(p.name, p.shape);
    });
    mm.add_instruction(migraphx::make_op(op_name, options), args);

    return mm == make_op_module("tm::" + op_name, options, mm.get_parameters());
}
} // namespace

TEST_CASE(torch_kit_common_unary_op_builder_test)
{
    const std::vector<std::string> unary_ops{
        "abs",  "acos",  "asin",    "atan",  "ceil", "cos",  "cosh",       "elu", "erf",
        "exp",  "floor", "isinf",   "isnan", "log",  "log2", "leaky_relu", "neg", "recip",
        "relu", "rsqrt", "sigmoid", "sign",  "sin",  "sinh", "sqrt",       "tan", "tanh"};

    std::for_each(unary_ops.begin(), unary_ops.end(), [&](const std::string& op_name) {
        migraphx::module mm;
        auto a = mm.add_parameter("a", {migraphx::shape::float_type, {3, 4}});
        add_common_op(mm, migraphx::make_op(op_name), {a});

        EXPECT(mm == make_op_module("tm::" + op_name, mm.get_parameters()));
    });
}

TEST_CASE(torch_kit_common_binary_op_builder_test)
{
    const std::vector<std::string> binary_ops{
        "add", "div", "equal", "fmod", "greater", "less", "max", "min", "mul", "pow", "sub"};

    std::for_each(binary_ops.begin(), binary_ops.end(), [&](const std::string& op_name) {
        migraphx::module mm;
        // Different ranks to exercise the common (numpy) broadcasting that the
        // common_ops wrapper inserts.
        auto a = mm.add_parameter("a", {migraphx::shape::float_type, {2, 3, 4}});
        auto b = mm.add_parameter("b", {migraphx::shape::float_type, {4}});
        add_common_op(mm, migraphx::make_op(op_name), {a, b});

        EXPECT(mm == make_op_module("tm::" + op_name, mm.get_parameters()));
    });
}

TEST_CASE(torch_kit_common_logical_op_builder_test)
{
    migraphx::module mm;
    auto a = mm.add_parameter("a", {migraphx::shape::bool_type, {2, 4}});
    auto b = mm.add_parameter("b", {migraphx::shape::bool_type, {2, 4}});
    add_common_op(mm, migraphx::make_op("logical_and"), {a, b});

    EXPECT(mm == make_op_module("tm::logical_and", mm.get_parameters()));
}

TEST_CASE(torch_kit_common_not_op_builder_test)
{
    migraphx::module mm;
    auto a = mm.add_parameter("a", {migraphx::shape::bool_type, {2, 4}});
    add_common_op(mm, migraphx::make_op("not"), {a});

    EXPECT(mm == make_op_module("tm::not", mm.get_parameters()));
}

TEST_CASE(torch_kit_common_bitwise_and_op_builder_test)
{
    // bitwise_and needs integral types; different ranks exercise common broadcasting.
    migraphx::module mm;
    auto a = mm.add_parameter("a", {migraphx::shape::int32_type, {2, 3, 4}});
    auto b = mm.add_parameter("b", {migraphx::shape::int32_type, {4}});
    add_common_op(mm, migraphx::make_op("bitwise_and"), {a, b});

    EXPECT(mm == make_op_module("tm::bitwise_and", mm.get_parameters()));
}

TEST_CASE(torch_kit_where_op_builder_test)
{
    // "where" is registered with common_type = false so the boolean condition is
    // not converted to the common type of the data operands.
    migraphx::module mm;
    auto cond = mm.add_parameter("cond", {migraphx::shape::bool_type, {2, 1}});
    auto a    = mm.add_parameter("a", {migraphx::shape::float_type, {2, 4}});
    auto b    = mm.add_parameter("b", {migraphx::shape::float_type, {2, 4}});
    add_common_op(mm, migraphx::make_op("where"), {cond, a, b}, {.common_type = false});

    EXPECT(mm == make_op_module("tm::where", mm.get_parameters()));
}

TEST_CASE(torch_kit_common_convert_op_builder_test)
{
    migraphx::value options{{"target_type", migraphx::shape::half_type}};

    migraphx::module mm;
    auto a = mm.add_parameter("a", {migraphx::shape::float_type, {3, 4}});
    add_common_op(mm, migraphx::make_op("convert", options), {a});

    EXPECT(mm == make_op_module("tm::convert", options, mm.get_parameters()));
}

TEST_CASE(torch_kit_ops_op_builder_test)
{
    // Every op in the kit's plain ops() list (mirrors torch_kit.cpp). Each builder
    // must insert exactly its wrapped op over the args, with the options untouched.
    const auto f   = migraphx::shape::float_type;
    const auto i64 = migraphx::shape::int64_type;
    const auto i8  = migraphx::shape::int8_type;
    const auto b   = migraphx::shape::bool_type;
    const auto obj = migraphx::value::object{};

    // A tuple input is needed to exercise get_tuple_elem.
    const migraphx::shape tuple_s{{migraphx::shape{f, {4, 6}}, migraphx::shape{f, {2, 3}}}};

    EXPECT(check_plain_op("argmax", {{"axis", 0}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("argmin", {{"axis", 0}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("broadcast", {{"axis", 1}, {"out_lens", {2, 4, 6}}}, {{"a", {f, {4}}}}));
    EXPECT(check_plain_op("concat", {{"axis", 0}}, {{"a", {f, {4, 6}}}, {"b", {f, {4, 6}}}}));
    EXPECT(check_plain_op("contiguous", obj, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op(
        "convolution_backwards", obj, {{"x", {f, {1, 3, 8, 8}}}, {"w", {f, {3, 4, 3, 3}}}}));
    EXPECT(check_plain_op("dequantizelinear", obj, {{"x", {i8, {4, 6}}}, {"scale", {f, {4, 6}}}}));
    EXPECT(check_plain_op("gather", {{"axis", 0}}, {{"data", {f, {4, 6}}}, {"ind", {i64, {2}}}}));
    EXPECT(check_plain_op("gathernd", obj, {{"data", {f, {4, 6}}}, {"ind", {i64, {2, 1}}}}));
    EXPECT(check_plain_op("get_tuple_elem", {{"index", 0}}, {{"a", tuple_s}}));
    EXPECT(check_plain_op("logsoftmax", {{"axis", 1}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("multibroadcast", {{"out_lens", {2, 4, 6}}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("pad", {{"pads", {0, 0, 1, 1}}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("pooling",
                          {{"mode", migraphx::op::pooling_mode::average}, {"lengths", {2, 2}}},
                          {{"a", {f, {1, 3, 8, 8}}}}));
    EXPECT(check_plain_op("prefix_scan_sum", {{"axis", 0}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("quantizelinear", obj, {{"x", {f, {4, 6}}}, {"scale", {f, {4, 6}}}}));
    EXPECT(check_plain_op("reduce_all", {{"axes", {1}}}, {{"a", {b, {4, 6}}}}));
    EXPECT(check_plain_op("reduce_any", {{"axes", {1}}}, {{"a", {b, {4, 6}}}}));
    EXPECT(check_plain_op("reduce_max", {{"axes", {1}}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("reduce_mean", {{"axes", {1}}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("reduce_min", {{"axes", {1}}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("reduce_prod", {{"axes", {1}}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("reduce_sum", {{"axes", {1}}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("reshape", {{"dims", {3, 8}}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("scatter_none",
                          {{"axis", 0}},
                          {{"data", {f, {4, 6}}}, {"ind", {i64, {2, 6}}}, {"upd", {f, {2, 6}}}}));
    EXPECT(check_plain_op(
        "slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {2}}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("softmax", {{"axis", 1}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("squeeze", {{"axes", {0}}}, {{"a", {f, {1, 4, 6}}}}));
    EXPECT(check_plain_op("step", {{"axes", {0}}, {"steps", {2}}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("topk", {{"k", 2}, {"axis", 0}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("transpose", {{"permutation", {1, 0}}}, {{"a", {f, {4, 6}}}}));
    EXPECT(check_plain_op("undefined", obj, {}));
    EXPECT(check_plain_op("unsqueeze", {{"axes", {0}}}, {{"a", {f, {4, 6}}}}));
}

TEST_CASE(torch_kit_unknown_op_builder_test)
{
    // The kit only registers under the "tm::" prefix; unknown names are not built.
    EXPECT(not migraphx::op::builder::has_op_builder("tm::not_a_real_op"));
}
