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

#include <onnx_test.hpp>
#include <migraphx/op/convolution.hpp>

// Input must always be of dimension 3 to pull batch, sequence and hidden size information
TEST_CASE(attention_invalid_input_dimensions)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_invalid_input_dimension.onnx"); }));
}

// We expect failure if the num_heads attribute is not set
TEST_CASE(attention_invalid_no_num_heads)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_invalid_no_num_heads.onnx"); }));
}

// Hidden sizes of weights must match that of input vector
TEST_CASE(attention_invalid_weight_hidden_size)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_invalid_weight_hidden_size.onnx"); }));
}

// Hidden sizes of weights if uneven must be defined by qkv
TEST_CASE(attention_invalid_weight_no_qkv_hidden_attr)
{
    EXPECT(
        test::throws([&] { optimize_onnx("attention_invalid_uneven_weight_no_qkv_hidden.onnx"); }));
}

// Bias dimensions must always be of size 1
TEST_CASE(attention_invalid_bias_dims_size)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_invalid_bias_dims_size.onnx"); }));
}

// Bias dimension value must always be equal to the sum of qkv sizes (3*hidden size if qkv aren't
// set)
TEST_CASE(attention_invalid_bias_value_size)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_invalid_bias_value_size.onnx"); }));
}

// Attention key pad mask (2d) must use (batch, sequence_length) as dimensions
TEST_CASE(attention_invalid_key_pad_mask)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_invalid_mask_2d_dims_test.onnx"); }));
}

// Attention key pad mask (3d) must use (batch, sequence_length, total_sequence_length) as
// dimensions
TEST_CASE(attention_invalid_key_pad_mask2)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_invalid_mask_3d_dims_test.onnx"); }));
}

// Attention 4D mask must use (batch, 1, sequence_length sequence_length) as dimensions
TEST_CASE(attention_invalid_4d_raw_mask)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_invalid_mask_4d_dims_test.onnx"); }));
}

// Attention 4D mask is (batch, 1, sequence_length sequence_length) thus last two dims must be equal
TEST_CASE(attention_invalid_4d_raw_mask2)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_invalid_mask_4d_last_dims_test.onnx"); }));
}

// Attention mask can have at most 5 dimensions
TEST_CASE(attention_invalid_5d_raw_mask)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_invalid_mask_5d_dims_test.onnx"); }));
}

// qkv_hidden_sizes attribute must be of dimension 3
TEST_CASE(attention_invalid_qkv_attr_dims)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_invalid_qkv_attr_test.onnx"); }));
}

// qkv_hidden_sizes attribute must be identical for Q and K values
TEST_CASE(attention_invalid_qkv_attr_values)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_invalid_qkv_attr_test2.onnx"); }));
}

// qkv_hidden_sizes attribute values my be greater than 0
TEST_CASE(attention_invalid_qkv_attr_values2)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_invalid_qkv_attr_test3.onnx"); }));
}

TEST_CASE(attention_invalid_input_num)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_invalid_input_num.onnx"); }));
}

TEST_CASE(attention_invalid_mask_type)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_invalid_mask_type_test.onnx"); }));
}
