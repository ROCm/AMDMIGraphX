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
#include <onnx_test_utils.hpp>

TEST_CASE(attention_double_head_bias_mask_test)
{
    // Batch 2, sequence length 2  num_heads 2, embedding_size  4
    //  Key pad masking and bias true
    migraphx::program p = make_attention_program(2, 2, 2, 4, true, true);
    auto prog           = optimize_onnx("attention_double_head_bias_mask_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(attention_double_head_3d_mask_unsupported)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_double_head_bias_3d_mask_test.onnx"); }));
}

TEST_CASE(attention_double_head_4d_mask_unsupported)
{
    EXPECT(test::throws([&] { optimize_onnx("attention_double_head_bias_4d_mask_test.onnx"); }));
}
