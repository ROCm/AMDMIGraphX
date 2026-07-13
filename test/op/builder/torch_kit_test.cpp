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

#include <limits>
#include <op_builder_test_utils.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/op/builder/op_builder.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/value.hpp>

// The torch_kit registers builders under the "tm::" prefix. The custom builder
// "tm::lstm" expands into an lstm op plus the rnn_last_hs_output and
// rnn_last_cell_output ops; the remaining builders are thin wrappers around
// native ops, either with common (broadcast/convert) handling or without.

namespace {
struct param_spec
{
    std::string name;
    migraphx::shape shape;
};

// Verifies that a plain (non-common) builder inserts exactly the wrapped op over
// the given args, unchanged. Builds the expected module by hand and compares it
// to what the kit's "tm::"-prefixed builder produces. Returns the comparison so
// the caller can EXPECT() it with the op name as a literal -- that way a failure
// message identifies which op did not match.
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

TEST_CASE(torch_lstm_forward_op_builder_test)
{
    const std::size_t hidden_size = 2;

    migraphx::module mm;
    auto x = mm.add_parameter("x", {migraphx::shape::float_type, {3, 4, 5}});
    auto w = mm.add_parameter("w", {migraphx::shape::float_type, {1, 8, 5}});
    auto r = mm.add_parameter("r", {migraphx::shape::float_type, {1, 8, 2}});

    // A forward lstm defaults to the {sigmoid, tanh, tanh} activation set.
    std::vector<migraphx::operation> actv_funcs{
        migraphx::make_op("sigmoid"), migraphx::make_op("tanh"), migraphx::make_op("tanh")};

    auto hs = mm.add_instruction(
        migraphx::make_op(
            "lstm", {{"hidden_size", hidden_size}, {"actv_func", migraphx::to_value(actv_funcs)}}),
        x,
        w,
        r);
    mm.add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
    mm.add_instruction(migraphx::make_op("rnn_last_cell_output"), hs);

    EXPECT(mm == make_op_module("tm::lstm", {{"hidden_size", hidden_size}}, mm.get_parameters()));
}

TEST_CASE(torch_lstm_bidirectional_op_builder_test)
{
    const std::size_t hidden_size = 2;

    migraphx::module mm;
    auto x = mm.add_parameter("x", {migraphx::shape::float_type, {3, 4, 5}});
    auto w = mm.add_parameter("w", {migraphx::shape::float_type, {2, 8, 5}});
    auto r = mm.add_parameter("r", {migraphx::shape::float_type, {2, 8, 2}});

    // A bidirectional lstm needs the activation set duplicated (6 functions).
    std::vector<migraphx::operation> actv_funcs{migraphx::make_op("sigmoid"),
                                                migraphx::make_op("tanh"),
                                                migraphx::make_op("tanh"),
                                                migraphx::make_op("sigmoid"),
                                                migraphx::make_op("tanh"),
                                                migraphx::make_op("tanh")};

    auto hs = mm.add_instruction(
        migraphx::make_op("lstm",
                          {{"hidden_size", hidden_size},
                           {"actv_func", migraphx::to_value(actv_funcs)},
                           {"direction", migraphx::op::rnn_direction::bidirectional}}),
        x,
        w,
        r);
    mm.add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
    mm.add_instruction(migraphx::make_op("rnn_last_cell_output"), hs);

    migraphx::value options{{"hidden_size", hidden_size},
                            {"direction", migraphx::op::rnn_direction::bidirectional}};
    EXPECT(mm == make_op_module("tm::lstm", options, mm.get_parameters()));
}

TEST_CASE(torch_lstm_custom_actv_funcs_op_builder_test)
{
    const std::size_t hidden_size = 2;

    // Explicitly provided activation functions should be used as-is and not be
    // overridden with the defaults.
    std::vector<migraphx::operation> actv_funcs{
        migraphx::make_op("tanh"), migraphx::make_op("sigmoid"), migraphx::make_op("sigmoid")};
    migraphx::value options{{"hidden_size", hidden_size},
                            {"actv_func", migraphx::to_value(actv_funcs)}};

    migraphx::module mm;
    auto x  = mm.add_parameter("x", {migraphx::shape::float_type, {3, 4, 5}});
    auto w  = mm.add_parameter("w", {migraphx::shape::float_type, {1, 8, 5}});
    auto r  = mm.add_parameter("r", {migraphx::shape::float_type, {1, 8, 2}});
    auto hs = mm.add_instruction(migraphx::make_op("lstm", options), x, w, r);
    mm.add_instruction(migraphx::make_op("rnn_last_hs_output"), hs);
    mm.add_instruction(migraphx::make_op("rnn_last_cell_output"), hs);

    EXPECT(mm == make_op_module("tm::lstm", options, mm.get_parameters()));
}

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

// tm::clip lowers to clip/min/max/identity based on which optional bounds are given
// (an undefined arg means "absent"); build_it shares a preamble so only the op differs.

TEST_CASE(torch_kit_clip_min_and_max_op_builder_test)
{
    const auto f  = migraphx::shape::float_type;
    auto build_it = [&](bool use_builder) {
        migraphx::module m;
        auto x  = m.add_parameter("x", {f, {2, 3}});
        auto lo = m.add_parameter("lo", {f, {2, 3}});
        auto hi = m.add_parameter("hi", {f, {2, 3}});
        if(use_builder)
            migraphx::op::builder::add("tm::clip", m, {x, lo, hi});
        else
            add_common_op(m, migraphx::make_op("clip"), {x, lo, hi});
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

TEST_CASE(torch_kit_clip_min_only_op_builder_test)
{
    // max is undefined -> lowers to max(x, lo).
    const auto f  = migraphx::shape::float_type;
    auto build_it = [&](bool use_builder) {
        migraphx::module m;
        auto x  = m.add_parameter("x", {f, {2, 3}});
        auto lo = m.add_parameter("lo", {f, {2, 3}});
        auto hi = m.add_instruction(migraphx::make_op("undefined"));
        if(use_builder)
            migraphx::op::builder::add("tm::clip", m, {x, lo, hi});
        else
            add_common_op(m, migraphx::make_op("max"), {x, lo});
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

TEST_CASE(torch_kit_clip_max_only_op_builder_test)
{
    // min is undefined -> lowers to min(x, hi).
    const auto f  = migraphx::shape::float_type;
    auto build_it = [&](bool use_builder) {
        migraphx::module m;
        auto x  = m.add_parameter("x", {f, {2, 3}});
        auto lo = m.add_instruction(migraphx::make_op("undefined"));
        auto hi = m.add_parameter("hi", {f, {2, 3}});
        if(use_builder)
            migraphx::op::builder::add("tm::clip", m, {x, lo, hi});
        else
            add_common_op(m, migraphx::make_op("min"), {x, hi});
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

TEST_CASE(torch_kit_clip_none_op_builder_test)
{
    // Neither bound supplied -> identity(x).
    const auto f  = migraphx::shape::float_type;
    auto build_it = [&](bool use_builder) {
        migraphx::module m;
        auto x  = m.add_parameter("x", {f, {2, 3}});
        auto lo = m.add_instruction(migraphx::make_op("undefined"));
        auto hi = m.add_instruction(migraphx::make_op("undefined"));
        if(use_builder)
            migraphx::op::builder::add("tm::clip", m, {x, lo, hi});
        else
            m.add_instruction(migraphx::make_op("identity"), x);
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

TEST_CASE(torch_kit_floor_div_op_builder_test)
{
    // tm::floor_div == floor(common div); different ranks exercise broadcasting.
    const auto f  = migraphx::shape::float_type;
    auto build_it = [&](bool use_builder) {
        migraphx::module m;
        auto a = m.add_parameter("a", {f, {2, 3, 4}});
        auto b = m.add_parameter("b", {f, {4}});
        if(use_builder)
            migraphx::op::builder::add("tm::floor_div", m, {a, b});
        else
        {
            auto quotient = add_common_op(m, migraphx::make_op("div"), {a, b});
            m.add_instruction(migraphx::make_op("floor"), quotient);
        }
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// tm::batchnorm is a thin re-export of the global "batchnorm" builder (running-stats
// batch norm: y = (x - mean) * rsqrt(var + eps) * scale + bias with channel-aligned
// params), so the "tm::"-prefixed form must match the un-prefixed builder exactly.
TEST_CASE(torch_kit_batchnorm_op_builder_test)
{
    const auto f = migraphx::shape::float_type;
    migraphx::value options{{"epsilon", 1e-5f}};
    auto build_it = [&](bool use_tm) {
        migraphx::module m;
        auto x     = m.add_parameter("x", {f, {2, 3, 4, 4}});
        auto scale = m.add_parameter("scale", {f, {3}});
        auto bias  = m.add_parameter("bias", {f, {3}});
        auto mean  = m.add_parameter("mean", {f, {3}});
        auto var   = m.add_parameter("var", {f, {3}});
        migraphx::op::builder::add(
            use_tm ? "tm::batchnorm" : "batchnorm", m, {x, scale, bias, mean, var}, options);
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// tm::layer_norm == (x - mean) * rsqrt(var + eps) * scale + bias, reduced over `axes`,
// with the affine params broadcast right-aligned against the input.
TEST_CASE(torch_kit_layer_norm_op_builder_test)
{
    const auto f              = migraphx::shape::float_type;
    const float eps           = 1e-5f;
    std::vector<int64_t> axes = {-1};
    auto build_it             = [&](bool use_builder) {
        migraphx::module m;
        auto x     = m.add_parameter("x", {f, {2, 3, 4}});
        auto scale = m.add_parameter("scale", {f, {4}});
        auto bias  = m.add_parameter("bias", {f, {4}});
        if(use_builder)
        {
            migraphx::op::builder::add(
                "tm::layer_norm", m, {x, scale, bias}, {{"epsilon", eps}, {"axes", axes}});
        }
        else
        {
            auto mean   = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), x);
            auto x_sub  = add_common_op(m, migraphx::make_op("sub"), {x, mean});
            auto sqdiff = add_common_op(m, migraphx::make_op("sqdiff"), {x, mean});
            auto variance =
                m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), sqdiff);
            auto eps_lit  = m.add_literal(migraphx::literal{migraphx::shape{f}, {eps}});
            auto var_eps  = add_common_op(m, migraphx::make_op("add"), {variance, eps_lit});
            auto rsqrt    = m.add_instruction(migraphx::make_op("rsqrt"), var_eps);
            auto norm     = add_common_op(m, migraphx::make_op("mul"), {x_sub, rsqrt});
            auto scaled   = add_common_op(m, migraphx::make_op("mul"), {norm, scale});
            add_common_op(m, migraphx::make_op("add"), {scaled, bias});
        }
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// tm::group_norm reshapes to (N, num_groups, -1), normalizes over the trailing axis,
// reshapes back, then applies the per-channel affine.
TEST_CASE(torch_kit_group_norm_op_builder_test)
{
    const auto f    = migraphx::shape::float_type;
    const float eps = 1e-5f;
    auto build_it   = [&](bool use_builder) {
        migraphx::module m;
        auto x     = m.add_parameter("x", {f, {2, 4, 3}});
        auto scale = m.add_parameter("scale", {f, {4}});
        auto bias  = m.add_parameter("bias", {f, {4}});
        if(use_builder)
        {
            migraphx::op::builder::add(
                "tm::group_norm", m, {x, scale, bias}, {{"epsilon", eps}, {"num_groups", 2}});
        }
        else
        {
            std::vector<int64_t> axes = {-1};
            auto grouped =
                m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 2, -1}}}), x);
            auto mean = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}),
                                          grouped);
            auto x_sub  = add_common_op(m, migraphx::make_op("sub"), {grouped, mean});
            auto sqdiff = add_common_op(m, migraphx::make_op("sqdiff"), {grouped, mean});
            auto variance =
                m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), sqdiff);
            auto eps_lit = m.add_literal(migraphx::literal{migraphx::shape{f}, {eps}});
            auto var_eps = add_common_op(m, migraphx::make_op("add"), {variance, eps_lit});
            auto rsqrt   = m.add_instruction(migraphx::make_op("rsqrt"), var_eps);
            auto norm    = add_common_op(m, migraphx::make_op("mul"), {x_sub, rsqrt});
            auto norm_r =
                m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 4, 3}}}), norm);
            auto scale_u =
                m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), scale);
            auto bias_u = m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), bias);
            auto scaled = add_common_op(m, migraphx::make_op("mul"), {norm_r, scale_u});
            add_common_op(m, migraphx::make_op("add"), {scaled, bias_u});
        }
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// tm::instance_norm computes stats from the input over the batch and spatial dims
// (every dim except channel dim 1), then applies the per-channel affine.
TEST_CASE(torch_kit_instance_norm_op_builder_test)
{
    const auto f    = migraphx::shape::float_type;
    const float eps = 1e-5f;
    auto build_it   = [&](bool use_builder) {
        migraphx::module m;
        auto x     = m.add_parameter("x", {f, {2, 3, 4, 4}});
        auto scale = m.add_parameter("scale", {f, {3}});
        auto bias  = m.add_parameter("bias", {f, {3}});
        if(use_builder)
        {
            migraphx::op::builder::add("tm::instance_norm", m, {x, scale, bias}, {{"epsilon", eps}});
        }
        else
        {
            std::vector<int64_t> axes = {0, 2, 3};
            auto mean   = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), x);
            auto x_sub  = add_common_op(m, migraphx::make_op("sub"), {x, mean});
            auto sqdiff = add_common_op(m, migraphx::make_op("sqdiff"), {x, mean});
            auto variance =
                m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", axes}}), sqdiff);
            auto eps_lit = m.add_literal(migraphx::literal{migraphx::shape{f}, {eps}});
            auto var_eps = add_common_op(m, migraphx::make_op("add"), {variance, eps_lit});
            auto rsqrt   = m.add_instruction(migraphx::make_op("rsqrt"), var_eps);
            auto norm    = add_common_op(m, migraphx::make_op("mul"), {x_sub, rsqrt});
            auto scale_u =
                m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), scale);
            auto bias_u =
                m.add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), bias);
            auto scaled = add_common_op(m, migraphx::make_op("mul"), {norm, scale_u});
            add_common_op(m, migraphx::make_op("add"), {scaled, bias_u});
        }
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// tm::vector_norm reduces abs(x) over axes with the ord-specific formula, then
// squeezes the reduced axes unless keepdim. General p-norm: sum(abs(x)^ord)^(1/ord).
TEST_CASE(torch_kit_vector_norm_p_op_builder_test)
{
    const auto f              = migraphx::shape::float_type;
    std::vector<int64_t> axes = {1};
    auto build_it             = [&](bool use_builder) {
        migraphx::module m;
        auto x = m.add_parameter("x", {f, {2, 3}});
        if(use_builder)
        {
            migraphx::op::builder::add(
                "tm::vector_norm", m, {x}, {{"ord", 2.0f}, {"axes", axes}, {"keepdim", false}});
        }
        else
        {
            auto abs_x   = m.add_instruction(migraphx::make_op("abs"), x);
            auto ord_lit = m.add_literal(migraphx::literal{migraphx::shape{f}, {2.0f}});
            auto pow_x   = add_common_op(m, migraphx::make_op("pow"), {abs_x, ord_lit});
            auto sum_pow =
                m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), pow_x);
            auto recip = m.add_instruction(migraphx::make_op("recip"), ord_lit);
            auto out   = add_common_op(m, migraphx::make_op("pow"), {sum_pow, recip});
            m.add_instruction(migraphx::make_op("squeeze", {{"axes", axes}}), out);
        }
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// ord = +inf -> max(abs(x)); keepdim = true leaves the reduced axis in place.
TEST_CASE(torch_kit_vector_norm_inf_op_builder_test)
{
    const auto f              = migraphx::shape::float_type;
    std::vector<int64_t> axes = {1};
    auto build_it             = [&](bool use_builder) {
        migraphx::module m;
        auto x = m.add_parameter("x", {f, {2, 3}});
        if(use_builder)
        {
            migraphx::op::builder::add(
                "tm::vector_norm",
                m,
                {x},
                {{"ord", std::numeric_limits<float>::infinity()}, {"axes", axes}, {"keepdim", true}});
        }
        else
        {
            auto abs_x = m.add_instruction(migraphx::make_op("abs"), x);
            m.add_instruction(migraphx::make_op("reduce_max", {{"axes", axes}}), abs_x);
        }
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// ord = 0 -> count of nonzero elements: sum(abs(x) > 0).
TEST_CASE(torch_kit_vector_norm_zero_op_builder_test)
{
    const auto f              = migraphx::shape::float_type;
    std::vector<int64_t> axes = {1};
    auto build_it             = [&](bool use_builder) {
        migraphx::module m;
        auto x = m.add_parameter("x", {f, {2, 3}});
        if(use_builder)
        {
            migraphx::op::builder::add(
                "tm::vector_norm", m, {x}, {{"ord", 0.0f}, {"axes", axes}, {"keepdim", false}});
        }
        else
        {
            auto abs_x   = m.add_instruction(migraphx::make_op("abs"), x);
            auto zero    = m.add_literal(migraphx::literal{migraphx::shape{f}, {0.0f}});
            auto nonzero = add_common_op(m, migraphx::make_op("greater"), {abs_x, zero});
            auto counts =
                m.add_instruction(migraphx::make_op("convert", {{"target_type", f}}), nonzero);
            auto out = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), counts);
            m.add_instruction(migraphx::make_op("squeeze", {{"axes", axes}}), out);
        }
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// tm::gelu_erf is a thin re-export of the global "gelu_erf" builder
// (0.5 * x * (1 + erf(x / sqrt(2)))), so the "tm::"-prefixed form must match it exactly.
TEST_CASE(torch_kit_gelu_erf_op_builder_test)
{
    const auto f  = migraphx::shape::float_type;
    auto build_it = [&](bool use_tm) {
        migraphx::module m;
        auto x = m.add_parameter("x", {f, {2, 3}});
        migraphx::op::builder::add(use_tm ? "tm::gelu_erf" : "gelu_erf", m, {x});
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// tm::glu splits the input in half along `axis` and gates the first half by
// sigmoid of the second: glu(x) = x1 * sigmoid(x2).
TEST_CASE(torch_kit_glu_op_builder_test)
{
    const auto f  = migraphx::shape::float_type;
    auto build_it = [&](bool use_builder) {
        migraphx::module m;
        auto x = m.add_parameter("x", {f, {2, 4}});
        if(use_builder)
        {
            migraphx::op::builder::add("tm::glu", m, {x}, {{"axis", -1}});
        }
        else
        {
            auto first = m.add_instruction(
                migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {2}}}), x);
            auto second = m.add_instruction(
                migraphx::make_op("slice", {{"axes", {1}}, {"starts", {2}}, {"ends", {4}}}), x);
            auto gate = m.add_instruction(migraphx::make_op("sigmoid"), second);
            add_common_op(m, migraphx::make_op("mul"), {first, gate});
        }
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// tm::selu == gamma * (max(0, x) + min(0, alpha * (exp(x) - 1))) with the SELU
// constants; literals are created in the builder's order so the modules match.
TEST_CASE(torch_kit_selu_op_builder_test)
{
    const auto f  = migraphx::shape::float_type;
    auto build_it = [&](bool use_builder) {
        migraphx::module m;
        auto x = m.add_parameter("x", {f, {2, 3}});
        if(use_builder)
        {
            migraphx::op::builder::add("tm::selu", m, {x});
        }
        else
        {
            auto zero     = m.add_literal(migraphx::literal{migraphx::shape{f}, {0.0f}});
            auto one      = m.add_literal(migraphx::literal{migraphx::shape{f}, {1.0f}});
            auto alpha    = m.add_literal(migraphx::literal{migraphx::shape{f},
                                                            {1.6732632423543772f}});
            auto gamma    = m.add_literal(migraphx::literal{migraphx::shape{f},
                                                            {1.0507009873554805f}});
            auto linear   = add_common_op(m, migraphx::make_op("max"), {zero, x});
            auto exp_x    = m.add_instruction(migraphx::make_op("exp"), x);
            auto exp_sub  = add_common_op(m, migraphx::make_op("sub"), {exp_x, one});
            auto exp_mul  = add_common_op(m, migraphx::make_op("mul"), {alpha, exp_sub});
            auto exp_part = add_common_op(m, migraphx::make_op("min"), {zero, exp_mul});
            auto sum      = add_common_op(m, migraphx::make_op("add"), {linear, exp_part});
            add_common_op(m, migraphx::make_op("mul"), {gamma, sum});
        }
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// tm::softsign == x / (1 + |x|).
TEST_CASE(torch_kit_softsign_op_builder_test)
{
    const auto f  = migraphx::shape::float_type;
    auto build_it = [&](bool use_builder) {
        migraphx::module m;
        auto x = m.add_parameter("x", {f, {2, 3}});
        if(use_builder)
        {
            migraphx::op::builder::add("tm::softsign", m, {x});
        }
        else
        {
            auto one   = m.add_literal(migraphx::literal{migraphx::shape{f}, {1.0f}});
            auto abs_x = m.add_instruction(migraphx::make_op("abs"), x);
            auto denom = add_common_op(m, migraphx::make_op("add"), {abs_x, one});
            add_common_op(m, migraphx::make_op("div"), {x, denom});
        }
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// tm::hardsigmoid == clip(alpha * x + beta, 0, 1) with alpha = 1/6, beta = 1/2.
TEST_CASE(torch_kit_hardsigmoid_op_builder_test)
{
    const auto f  = migraphx::shape::float_type;
    auto build_it = [&](bool use_builder) {
        migraphx::module m;
        auto x = m.add_parameter("x", {f, {2, 3}});
        if(use_builder)
        {
            migraphx::op::builder::add("tm::hardsigmoid", m, {x});
        }
        else
        {
            auto alpha   = m.add_literal(migraphx::literal{migraphx::shape{f}, {1.0f / 6.0f}});
            auto beta    = m.add_literal(migraphx::literal{migraphx::shape{f}, {0.5f}});
            auto lo      = m.add_literal(migraphx::literal{migraphx::shape{f}, {0.0f}});
            auto hi      = m.add_literal(migraphx::literal{migraphx::shape{f}, {1.0f}});
            auto scaled  = add_common_op(m, migraphx::make_op("mul"), {alpha, x});
            auto shifted = add_common_op(m, migraphx::make_op("add"), {beta, scaled});
            add_common_op(m, migraphx::make_op("clip"), {shifted, lo, hi});
        }
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// tm::nan_to_num replaces NaN with `nan`, +inf with `posinf`, -inf with `neginf`;
// the inf sign is recovered by comparing the input against 0. where broadcasts its
// operands but does not promote the boolean condition.
TEST_CASE(torch_kit_nan_to_num_op_builder_test)
{
    const auto f = migraphx::shape::float_type;
    migraphx::value options{{"nan", 0.0f}, {"posinf", 1e4f}, {"neginf", -1e4f}};
    auto build_it = [&](bool use_builder) {
        migraphx::module m;
        auto x = m.add_parameter("x", {f, {2, 3}});
        if(use_builder)
        {
            migraphx::op::builder::add("tm::nan_to_num", m, {x}, options);
        }
        else
        {
            auto nan_lit    = m.add_literal(migraphx::literal{migraphx::shape{f}, {0.0f}});
            auto zero       = m.add_literal(migraphx::literal{migraphx::shape{f}, {0.0f}});
            auto posinf_lit = m.add_literal(migraphx::literal{migraphx::shape{f}, {1e4f}});
            auto neginf_lit = m.add_literal(migraphx::literal{migraphx::shape{f}, {-1e4f}});

            auto is_nan = m.add_instruction(migraphx::make_op("isnan"), x);
            auto result = add_common_op(
                m, migraphx::make_op("where"), {is_nan, nan_lit, x}, {.common_type = false});
            auto is_inf   = m.add_instruction(migraphx::make_op("isinf"), x);
            auto less     = add_common_op(m, migraphx::make_op("less"), {x, zero});
            auto greater  = add_common_op(m, migraphx::make_op("greater"), {x, zero});
            auto neg_mask = add_common_op(m, migraphx::make_op("logical_and"), {less, is_inf});
            auto pos_mask = add_common_op(m, migraphx::make_op("logical_and"), {greater, is_inf});
            result        = add_common_op(m,
                                   migraphx::make_op("where"),
                                   {neg_mask, neginf_lit, result},
                                   {.common_type = false});
            add_common_op(m,
                          migraphx::make_op("where"),
                          {pos_mask, posinf_lit, result},
                          {.common_type = false});
        }
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// tm::matmul reuses the shared dot builder, so tm::dot must alias it (numpy
// batch-broadcast + dot). Mixed batch ranks exercise the broadcasting.
TEST_CASE(torch_kit_dot_op_builder_test)
{
    const auto f  = migraphx::shape::float_type;
    auto build_it = [&](bool prefixed) {
        migraphx::module m;
        auto a = m.add_parameter("a", {f, {2, 1, 3, 4}});
        auto b = m.add_parameter("b", {f, {5, 4, 6}});
        migraphx::op::builder::add(prefixed ? "tm::dot" : "dot", m, {a, b});
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// ND linear flattens to rank 2, delegates to gemm, then reshapes back.
TEST_CASE(torch_kit_linear_op_builder_test)
{
    const auto f  = migraphx::shape::float_type;
    auto build_it = [&](bool use_builder) {
        migraphx::module m;
        auto x    = m.add_parameter("x", {f, {2, 3, 4}});
        auto w    = m.add_parameter("w", {f, {5, 4}});
        auto bias = m.add_parameter("bias", {f, {5}});
        if(use_builder)
        {
            migraphx::op::builder::add("tm::linear", m, {x, w, bias});
        }
        else
        {
            auto x2d = m.add_instruction(migraphx::make_op("reshape", {{"dims", {6, 4}}}), x);
            auto out =
                migraphx::op::builder::add("gemm", m, {x2d, w, bias}, {{"transB", true}}).front();
            m.add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 5}}}), out);
        }
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// rank-2 linear is exactly the gemm builder.
TEST_CASE(torch_kit_linear_no_bias_op_builder_test)
{
    const auto f  = migraphx::shape::float_type;
    auto build_it = [&](bool use_builder) {
        migraphx::module m;
        auto x = m.add_parameter("x", {f, {3, 4}});
        auto w = m.add_parameter("w", {f, {5, 4}});
        if(use_builder)
            migraphx::op::builder::add("tm::linear", m, {x, w});
        else
            migraphx::op::builder::add("gemm", m, {x, w}, {{"transB", true}});
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// conv reuses the shared convolution builder (conv + fused channel bias), so
// tm::convolution must alias it. Note the builder's plural attribute names.
TEST_CASE(torch_kit_convolution_op_builder_test)
{
    const auto f                       = migraphx::shape::float_type;
    std::vector<std::size_t> strides   = {1, 1};
    std::vector<std::size_t> paddings  = {0, 0};
    std::vector<std::size_t> dilations = {1, 1};
    migraphx::value options{
        {"strides", strides}, {"paddings", paddings}, {"dilations", dilations}, {"group", 1}};
    auto build_it = [&](bool prefixed) {
        migraphx::module m;
        auto x    = m.add_parameter("x", {f, {1, 3, 8, 8}});
        auto w    = m.add_parameter("w", {f, {4, 3, 3, 3}});
        auto bias = m.add_parameter("bias", {f, {4}});
        migraphx::op::builder::add(
            prefixed ? "tm::convolution" : "convolution", m, {x, w, bias}, options);
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// tm::conv_transpose runs convolution_backwards unpadded, crops off the symmetric
// padding while keeping the output_padding elements, then adds the channel bias.
TEST_CASE(torch_kit_conv_transpose_op_builder_test)
{
    const auto f                           = migraphx::shape::float_type;
    std::vector<std::size_t> stride         = {2, 2};
    std::vector<std::size_t> padding        = {1, 1};
    std::vector<std::size_t> dilation       = {1, 1};
    std::vector<std::size_t> output_padding = {1, 1};
    migraphx::value options{{"stride", stride},
                            {"padding", padding},
                            {"dilation", dilation},
                            {"output_padding", output_padding},
                            {"group", 1}};
    auto build_it = [&](bool use_builder) {
        migraphx::module m;
        auto x    = m.add_parameter("x", {f, {1, 3, 4, 4}});
        auto w    = m.add_parameter("w", {f, {3, 4, 3, 3}});
        auto bias = m.add_parameter("bias", {f, {4}});
        if(use_builder)
        {
            migraphx::op::builder::add("tm::conv_transpose", m, {x, w, bias}, options);
        }
        else
        {
            auto out = m.add_instruction(
                migraphx::make_op(
                    "convolution_backwards",
                    {{"stride", stride}, {"padding", {0, 0}}, {"dilation", dilation}, {"group", 1}}),
                x,
                w);
            auto cropped = m.add_instruction(
                migraphx::make_op("slice",
                                  {{"axes", {2, 3}}, {"starts", {1, 1}}, {"ends", {9, 9}}}),
                out);
            auto b = m.add_instruction(
                migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 4, 8, 8}}}), bias);
            m.add_instruction(migraphx::make_op("add"), cropped, b);
        }
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// tm::conv_transpose with no output_padding passes padding straight to the op.
TEST_CASE(torch_kit_conv_transpose_no_crop_op_builder_test)
{
    const auto f                           = migraphx::shape::float_type;
    std::vector<std::size_t> stride         = {1, 1};
    std::vector<std::size_t> padding        = {1, 1};
    std::vector<std::size_t> dilation       = {1, 1};
    std::vector<std::size_t> output_padding = {0, 0};
    migraphx::value options{{"stride", stride},
                            {"padding", padding},
                            {"dilation", dilation},
                            {"output_padding", output_padding},
                            {"group", 1}};
    auto build_it = [&](bool use_builder) {
        migraphx::module m;
        auto x = m.add_parameter("x", {f, {1, 3, 4, 4}});
        auto w = m.add_parameter("w", {f, {3, 4, 3, 3}});
        if(use_builder)
        {
            migraphx::op::builder::add("tm::conv_transpose", m, {x, w}, options);
        }
        else
        {
            m.add_instruction(
                migraphx::make_op(
                    "convolution_backwards",
                    {{"stride", stride}, {"padding", padding}, {"dilation", dilation}, {"group", 1}}),
                x,
                w);
        }
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

// tm::std == sqrt(sum((x - mean)^2) / (N - correction)) reduced over the axes,
// squeezing them out unless keepdim. Here N == 4 and correction == 1, so N - 1 == 3.
TEST_CASE(torch_kit_std_op_builder_test)
{
    const auto f = migraphx::shape::float_type;
    migraphx::value options{{"axes", {1}}, {"keepdim", false}, {"correction", 1.0f}};
    auto build_it = [&](bool use_builder) {
        migraphx::module m;
        auto x = m.add_parameter("x", {f, {2, 4}});
        if(use_builder)
        {
            migraphx::op::builder::add("tm::std", m, {x}, options);
        }
        else
        {
            auto mean = m.add_instruction(migraphx::make_op("reduce_mean", {{"axes", {1}}}), x);
            auto sub  = add_common_op(m, migraphx::make_op("sub"), {x, mean});
            auto sq   = add_common_op(m, migraphx::make_op("mul"), {sub, sub});
            auto sum  = m.add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sq);
            auto denom = m.add_literal(migraphx::literal{migraphx::shape{f}, {3.0f}});
            auto var   = add_common_op(m, migraphx::make_op("div"), {sum, denom});
            auto out   = m.add_instruction(migraphx::make_op("sqrt"), var);
            m.add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), out);
        }
        return m;
    };
    EXPECT(build_it(true) == build_it(false));
}

TEST_CASE(torch_kit_unknown_op_builder_test)
{
    // The kit only registers under the "tm::" prefix; unknown names are not built.
    EXPECT(not migraphx::op::builder::has_op_builder("tm::not_a_real_op"));
}
