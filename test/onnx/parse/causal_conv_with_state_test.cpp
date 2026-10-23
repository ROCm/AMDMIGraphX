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
#include <limits>

static migraphx::operation depthwise_conv(int64_t channels, int64_t dilation = 1)
{
    return migraphx::make_op(
        "convolution",
        {{"padding", {0}}, {"stride", {1}}, {"dilation", {dilation}}, {"group", channels}});
}

// The last `state_length` positions along `axis`
static migraphx::operation state_tail(int64_t axis, int64_t state_length)
{
    return migraphx::make_op("slice",
                             {{"axes", {axis}},
                              {"starts", {-state_length}},
                              {"ends", {std::numeric_limits<int64_t>::max()}}});
}

TEST_CASE(causal_conv_with_state_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto input  = mm->add_parameter("input", {migraphx::shape::float_type, {1, 2, 4}});
    auto weight = mm->add_parameter("weight", {migraphx::shape::float_type, {2, 1, 3}});
    auto bias   = mm->add_parameter("bias", {migraphx::shape::float_type, {2}});
    auto past   = mm->add_parameter("past_state", {migraphx::shape::float_type, {1, 2, 2}});
    auto seq    = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), past, input);
    auto conv   = mm->add_instruction(depthwise_conv(2), seq, weight);
    auto bias_b = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 2, 4}}}), bias);
    auto add     = mm->add_instruction(migraphx::make_op("add"), conv, bias_b);
    auto sigmoid = mm->add_instruction(migraphx::make_op("sigmoid"), add);
    auto output  = mm->add_instruction(migraphx::make_op("mul"), add, sigmoid);
    auto present = mm->add_instruction(state_tail(2, 2), seq);
    mm->add_return({output, present});

    auto prog = read_onnx("causal_conv_with_state_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(causal_conv_with_state_no_state_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto input  = mm->add_parameter("input", {migraphx::shape::float_type, {1, 2, 4}});
    auto weight = mm->add_parameter("weight", {migraphx::shape::float_type, {2, 1, 3}});
    auto seq = mm->add_instruction(migraphx::make_op("pad", {{"pads", {0, 0, 2, 0, 0, 0}}}), input);
    auto output = mm->add_instruction(depthwise_conv(2), seq, weight);
    mm->add_return({output});

    auto prog = read_onnx("causal_conv_with_state_no_state_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(causal_conv_with_state_dyn_batch_test)
{
    EXPECT(check_parse("causal_conv_with_state_no_state_test.onnx",
                       {{"input", {migraphx::shape::float_type, {{1, 4}, {2, 2}, {4, 4}}}},
                        {"weight", {migraphx::shape::float_type, {2, 1, 3}}}},
                       [](migraphx::module& m, const auto& args) {
                           auto seq = m.add_instruction(
                               migraphx::make_op("pad", {{"pads", {0, 0, 2, 0, 0, 0}}}), args[0]);
                           auto output = m.add_instruction(depthwise_conv(2), seq, args[1]);
                           m.add_return({output});
                       }));
}

TEST_CASE(causal_conv_with_state_channels_last_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto input  = mm->add_parameter("input", {migraphx::shape::float_type, {1, 3, 2, 2}});
    auto weight = mm->add_parameter("weight", {migraphx::shape::float_type, {4, 1, 3}});
    auto bias   = mm->add_parameter("bias", {migraphx::shape::float_type, {4}});
    auto past   = mm->add_parameter("past_state", {migraphx::shape::float_type, {1, 2, 2, 2}});
    auto seq    = mm->add_instruction(migraphx::make_op("concat", {{"axis", 1}}), past, input);
    auto x      = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {0, 0, -1}}}), seq);
    x = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), x);
    auto conv   = mm->add_instruction(depthwise_conv(4), x, weight);
    auto bias_b = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {1, 4, 3}}}), bias);
    auto add     = mm->add_instruction(migraphx::make_op("add"), conv, bias_b);
    auto sigmoid = mm->add_instruction(migraphx::make_op("sigmoid"), add);
    auto output  = mm->add_instruction(migraphx::make_op("mul"), add, sigmoid);
    output =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), output);
    output = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {0, 0, 2, 2}}}), output);
    auto present = mm->add_instruction(state_tail(1, 2), seq);
    mm->add_return({output, present});

    auto prog = read_onnx("causal_conv_with_state_channels_last_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(causal_conv_with_state_window_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto input  = mm->add_parameter("input", {migraphx::shape::float_type, {1, 2, 2}});
    auto weight = mm->add_parameter("weight", {migraphx::shape::float_type, {2, 1, 3}});
    auto past   = mm->add_parameter("past_state", {migraphx::shape::float_type, {3, 1, 2, 2}});
    mm->add_instruction(migraphx::make_op("undefined"));
    // Only the last window slot is the state after the last position
    auto last = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), past);
    last        = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), last);
    auto seq    = mm->add_instruction(migraphx::make_op("concat", {{"axis", 2}}), last, input);
    auto output = mm->add_instruction(depthwise_conv(2), seq, weight);
    // Three slots for two positions: the first slot predates this call and is zero
    auto zero =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {0.0f}});
    auto zeros = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 2, 2}}}), zero);
    auto slot1 = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2}}, {"starts", {-3}}, {"ends", {-1}}}), seq);
    slot1      = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), slot1);
    auto slot2 = mm->add_instruction(state_tail(2, 2), seq);
    slot2      = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), slot2);
    auto present =
        mm->add_instruction(migraphx::make_op("concat", {{"axis", 0}}), zeros, slot1, slot2);
    mm->add_return({output, present});

    auto prog = read_onnx("causal_conv_with_state_window_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(causal_conv_with_state_activation_error_test)
{
    EXPECT(test::throws([&] { read_onnx("causal_conv_with_state_activation_error_test.onnx"); }));
}

TEST_CASE(causal_conv_with_state_ndim_error_test)
{
    EXPECT(test::throws([&] { read_onnx("causal_conv_with_state_ndim_error_test.onnx"); }));
}

TEST_CASE(causal_conv_with_state_dilation_error_test)
{
    EXPECT(test::throws([&] { read_onnx("causal_conv_with_state_dilation_error_test.onnx"); }));
}

TEST_CASE(causal_conv_with_state_window_error_test)
{
    EXPECT(test::throws([&] { read_onnx("causal_conv_with_state_window_error_test.onnx"); }));
}

TEST_CASE(causal_conv_with_state_weight_error_test)
{
    EXPECT(test::throws([&] { read_onnx("causal_conv_with_state_weight_error_test.onnx"); }));
}

TEST_CASE(causal_conv_with_state_bias_error_test)
{
    EXPECT(test::throws([&] { read_onnx("causal_conv_with_state_bias_error_test.onnx"); }));
}

TEST_CASE(causal_conv_with_state_past_state_error_test)
{
    EXPECT(test::throws([&] { read_onnx("causal_conv_with_state_past_state_error_test.onnx"); }));
}
