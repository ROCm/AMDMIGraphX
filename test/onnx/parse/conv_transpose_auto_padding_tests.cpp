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

// Test 1: SAME_UPPER with symmetric padding (odd kernel)
TEST_CASE(conv_transpose_auto_pad_same_upper_symmetric_test)
{
    // With kernel=3, stride=1, the padding is symmetric: (1, 1)
    // SAME_UPPER and SAME_LOWER produce the same result
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto input  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto weight = mm->add_parameter("w", {migraphx::shape::float_type, {1, 1, 3, 3}});

    // Symmetric padding: both dimensions get padding=1
    auto conv_transpose = mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {1, 1}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
        input,
        weight);
    mm->add_return({conv_transpose});

    auto prog = read_onnx("conv_transpose_auto_pad_same_upper_symmetric_test.onnx");
    EXPECT(p == prog);
}

// Test 2: SAME_UPPER with asymmetric padding (even kernel)
TEST_CASE(conv_transpose_auto_pad_same_upper_asymmetric_test)
{
    // With kernel=4 (even), stride=1, total padding needed = 3
    // SAME_UPPER splits as: left=1, right=2 (extra padding on the right/bottom)
    // This creates asymmetric padding
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto input  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 4, 4}});
    auto weight = mm->add_parameter("w", {migraphx::shape::float_type, {1, 1, 4, 4}});

    // Asymmetric padding handled by: padding set to 0, then slice to crop output
    auto conv_transpose = mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
        input,
        weight);

    // Slice to handle asymmetric padding: starts=[1, 1], ends=[5, 5]
    // This crops the output from [1, 1, 7, 7] to [1, 1, 4, 4]
    auto sliced = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2, 3}}, {"starts", {1, 1}}, {"ends", {5, 5}}}),
        conv_transpose);
    mm->add_return({sliced});

    auto prog = read_onnx("conv_transpose_auto_pad_same_upper_asymmetric_test.onnx");
    EXPECT(p == prog);
}

// Test 3: SAME_LOWER with asymmetric padding (even kernel)
TEST_CASE(conv_transpose_auto_pad_same_lower_asymmetric_test)
{
    // With kernel=4 (even), stride=1, total padding needed = 3
    // SAME_LOWER splits as: left=2, right=1 (extra padding on the left/top)
    // This is DIFFERENT from SAME_UPPER
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto input  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 4, 4}});
    auto weight = mm->add_parameter("w", {migraphx::shape::float_type, {1, 1, 4, 4}});

    // Asymmetric padding handled by: padding set to 0, then slice
    auto conv_transpose = mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
        input,
        weight);

    // Slice for SAME_LOWER: starts=[2, 2], ends=[6, 6]
    // Different from SAME_UPPER! More padding removed from the start
    auto sliced = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2, 3}}, {"starts", {2, 2}}, {"ends", {6, 6}}}),
        conv_transpose);
    mm->add_return({sliced});

    auto prog = read_onnx("conv_transpose_auto_pad_same_lower_asymmetric_test.onnx");
    EXPECT(p == prog);
}

// Test 4: VALID (no padding)
TEST_CASE(conv_transpose_auto_pad_valid_test)
{
    // VALID means no padding at all
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto input  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 3, 3}});
    auto weight = mm->add_parameter("w", {migraphx::shape::float_type, {1, 1, 3, 3}});

    // No padding applied
    auto conv_transpose = mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {0, 0}}, {"stride", {1, 1}}, {"dilation", {1, 1}}}),
        input,
        weight);
    mm->add_return({conv_transpose});

    auto prog = read_onnx("conv_transpose_auto_pad_valid_test.onnx");
    EXPECT(p == prog);
}

// Test 5: SAME_UPPER with strides > 1
TEST_CASE(conv_transpose_auto_pad_same_upper_stride_test)
{
    // Test auto_pad with non-unit strides
    // With kernel=3, stride=2, input=4, expected output = input * stride = 8
    // total_padding needed = kernel - stride = 3 - 2 = 1 (asymmetric)
    // SAME_UPPER: left=0, right=1
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto input  = mm->add_parameter("x", {migraphx::shape::float_type, {1, 1, 4, 4}});
    auto weight = mm->add_parameter("w", {migraphx::shape::float_type, {1, 1, 3, 3}});

    // Asymmetric padding: use padding=0 and slice to handle it
    auto conv_transpose = mm->add_instruction(
        migraphx::make_op("convolution_backwards",
                          {{"padding", {0, 0}}, {"stride", {2, 2}}, {"dilation", {1, 1}}}),
        input,
        weight);

    // Slice to crop from [1, 1, 9, 9] to [1, 1, 8, 8]
    // starts=[0, 0], ends=[8, 8]
    auto sliced = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2, 3}}, {"starts", {0, 0}}, {"ends", {8, 8}}}),
        conv_transpose);
    mm->add_return({sliced});

    auto prog = read_onnx("conv_transpose_auto_pad_same_upper_stride_test.onnx");
    EXPECT(p == prog);
}
