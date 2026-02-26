/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "migraphx/make_op.hpp"
#include <onnx_test.hpp>

TEST_CASE(matmulbnb4_fp4_test)
{
    // Test MatMulBnb4 with FP4 quantization (quant_type=0)
    // N=4, K=8, block_size=16
    // Input A: [2, 8], B (packed): [16], absmax: [2]
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto a      = mm->add_parameter("A", migraphx::shape{migraphx::shape::float_type, {2, 8}});
    auto b      = mm->add_parameter("B", migraphx::shape{migraphx::shape::uint8_type, {16}});
    auto absmax = mm->add_parameter("absmax", migraphx::shape{migraphx::shape::float_type, {2}});

    // Unpack 4-bit data and reshape to (N, K) = (4, 8)
    auto unpacked_b = mm->add_instruction(migraphx::make_op("unpack_int4"), b);
    unpacked_b = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 8}}}), unpacked_b);

    // Prepare absmax for blockwise dequantization
    // absmax: [2] -> [2, 1] -> [2, 16] -> [32] -> [4, 8]
    auto expanded_absmax =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), absmax);
    expanded_absmax = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {2, 16}}}), expanded_absmax);
    expanded_absmax =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {32}}}), expanded_absmax);
    expanded_absmax =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 8}}}), expanded_absmax);

    // Dequantize: convert to float, divide by scale factor (8 for FP4), multiply by absmax
    auto float_data = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), unpacked_b);
    auto scale_factor =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {8.0f}});
    auto scale_factor_bc = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {4, 8}}}), scale_factor);
    auto scaled_data = mm->add_instruction(migraphx::make_op("div"), float_data, scale_factor_bc);
    auto dequantized = mm->add_instruction(migraphx::make_op("mul"), scaled_data, expanded_absmax);

    // Transpose dequantized B from (N, K) to (K, N)
    dequantized =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dequantized);

    // Perform matmul: A [2, 8] x B [8, 4] -> Y [2, 4]
    mm->add_instruction(migraphx::make_op("dot"), a, dequantized);

    auto prog = optimize_onnx("matmulbnb4_fp4_test.onnx");

    p.sort();
    prog.sort();
    EXPECT(p == prog);
}

TEST_CASE(matmulbnb4_nf4_test)
{
    // Test MatMulBnb4 with NF4 quantization (quant_type=1)
    // N=8, K=16, block_size=16
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto a      = mm->add_parameter("A", migraphx::shape{migraphx::shape::float_type, {3, 16}});
    auto b      = mm->add_parameter("B", migraphx::shape{migraphx::shape::uint8_type, {64}});
    auto absmax = mm->add_parameter("absmax", migraphx::shape{migraphx::shape::float_type, {8}});

    // Unpack and reshape
    auto unpacked_b = mm->add_instruction(migraphx::make_op("unpack_int4"), b);
    unpacked_b = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {8, 16}}}), unpacked_b);

    // Prepare absmax
    auto expanded_absmax =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), absmax);
    expanded_absmax = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {8, 16}}}), expanded_absmax);
    expanded_absmax =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {128}}}), expanded_absmax);
    expanded_absmax =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {8, 16}}}), expanded_absmax);

    // Dequantize (NF4 uses same scale factor for simplified implementation)
    auto float_data = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), unpacked_b);
    auto scale_factor =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {8.0f}});
    auto scale_factor_bc = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {8, 16}}}), scale_factor);
    auto scaled_data = mm->add_instruction(migraphx::make_op("div"), float_data, scale_factor_bc);
    auto dequantized = mm->add_instruction(migraphx::make_op("mul"), scaled_data, expanded_absmax);

    // Transpose
    dequantized =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dequantized);

    // Matmul
    mm->add_instruction(migraphx::make_op("dot"), a, dequantized);

    auto prog = optimize_onnx("matmulbnb4_nf4_test.onnx");

    p.sort();
    prog.sort();
    EXPECT(p == prog);
}

// Error test cases
TEST_CASE(matmulbnb4_invalid_quant_type_test)
{
    // Should throw error: quant_type must be 0 (FP4) or 1 (NF4)
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_invalid_quant_type_test.onnx"); }));
}

TEST_CASE(matmulbnb4_invalid_block_size_test)
{
    // Should throw error: block_size must be power of 2 and >= 16
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_invalid_block_size_test.onnx"); }));
}

TEST_CASE(matmulbnb4_invalid_block_size_small_test)
{
    // Should throw error: block_size must be >= 16
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_invalid_block_size_small_test.onnx"); }));
}

TEST_CASE(matmulbnb4_wrong_input_count_test)
{
    // Should throw error: requires exactly 3 inputs
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_wrong_input_count_test.onnx"); }));
}

TEST_CASE(matmulbnb4_wrong_a_dims_test)
{
    // Should throw error: Input A must have at least 2 dimensions
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_wrong_a_dims_test.onnx"); }));
}

TEST_CASE(matmulbnb4_wrong_a_inner_dim_test)
{
    // Should throw error: Input A inner dimension must match attribute K
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_wrong_a_inner_dim_test.onnx"); }));
}

TEST_CASE(matmulbnb4_wrong_b_dims_test)
{
    // Should throw error: Input B does not match expected dimensions
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_wrong_b_dims_test.onnx"); }));
}

TEST_CASE(matmulbnb4_wrong_absmax_dims_test)
{
    // Should throw error: Input absmax does not match expected dimensions
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_wrong_absmax_dims_test.onnx"); }));
}

TEST_CASE(matmulbnb4_missing_n_attr_test)
{
    // Should throw error: Missing required attribute N
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_missing_n_attr_test.onnx"); }));
}
