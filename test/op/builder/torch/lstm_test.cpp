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
#include <migraphx/operation.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/value.hpp>

// The tm::lstm builder expands into an lstm op plus the rnn_last_hs_output and
// rnn_last_cell_output ops.

TEST_CASE(torch_kit_lstm_forward_op_builder_test)
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

TEST_CASE(torch_kit_lstm_bidirectional_op_builder_test)
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

TEST_CASE(torch_kit_lstm_custom_actv_funcs_op_builder_test)
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
