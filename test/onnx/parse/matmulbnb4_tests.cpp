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

#include <migraphx/make_op.hpp>
#include <onnx_test.hpp>

TEST_CASE(matmulbnb4_fp4_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto a      = mm->add_parameter("A", migraphx::shape{migraphx::shape::float_type, {2, 8}});
    auto b      = mm->add_parameter("B", migraphx::shape{migraphx::shape::uint8_type, {16}});
    auto absmax = mm->add_parameter("absmax", migraphx::shape{migraphx::shape::float_type, {2}});

    auto unpacked_b = mm->add_instruction(migraphx::make_op("unpack_int4"), b);
    unpacked_b = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 8}}}), unpacked_b);

    auto expanded_absmax =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), absmax);
    expanded_absmax = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {2, 16}}}), expanded_absmax);
    expanded_absmax =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {32}}}), expanded_absmax);
    expanded_absmax =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 8}}}), expanded_absmax);

    auto float_data = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), unpacked_b);
    auto scale_factor =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {8.0f}});
    auto scale_factor_bc = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {4, 8}}}), scale_factor);
    auto scaled_data = mm->add_instruction(migraphx::make_op("div"), float_data, scale_factor_bc);
    auto dequantized = mm->add_instruction(migraphx::make_op("mul"), scaled_data, expanded_absmax);

    dequantized =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dequantized);

    mm->add_instruction(migraphx::make_op("dot"), a, dequantized);

    auto prog = optimize_onnx("matmulbnb4_fp4_test.onnx");

    p.sort();
    prog.sort();
    EXPECT(p == prog);
}

TEST_CASE(matmulbnb4_nf4_test)
{
    std::vector<float> nf4_lookup_table = {-1.0f,
                                           -0.6961928009986877f,
                                           -0.5250730514526367f,
                                           -0.39491748809814453f,
                                           -0.28444138169288635f,
                                           -0.18477343022823334f,
                                           -0.09105003625154495f,
                                           0.0f,
                                           0.07958029955625534f,
                                           0.16093020141124725f,
                                           0.24611230194568634f,
                                           0.33791524171829224f,
                                           0.44070982933044434f,
                                           0.5626170039176941f,
                                           0.7229568362236023f,
                                           1.0f};

    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto a      = mm->add_parameter("A", migraphx::shape{migraphx::shape::float_type, {3, 16}});
    auto b      = mm->add_parameter("B", migraphx::shape{migraphx::shape::uint8_type, {64}});
    auto absmax = mm->add_parameter("absmax", migraphx::shape{migraphx::shape::float_type, {8}});

    auto unpacked_b = mm->add_instruction(migraphx::make_op("unpack_int4"), b);
    unpacked_b = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {8, 16}}}), unpacked_b);

    auto expanded_absmax =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), absmax);
    expanded_absmax = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {8, 16}}}), expanded_absmax);
    expanded_absmax =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {128}}}), expanded_absmax);
    expanded_absmax =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {8, 16}}}), expanded_absmax);

    auto lut = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::float_type, {16}}, nf4_lookup_table});
    auto indices = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int64_type}}), unpacked_b);
    auto dequant_values =
        mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), lut, indices);
    auto dequantized =
        mm->add_instruction(migraphx::make_op("mul"), dequant_values, expanded_absmax);

    dequantized =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dequantized);

    mm->add_instruction(migraphx::make_op("dot"), a, dequantized);

    auto prog = optimize_onnx("matmulbnb4_nf4_test.onnx");

    p.sort();
    prog.sort();
    EXPECT(p == prog);
}

TEST_CASE(matmulbnb4_fp4_non_aligned_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto a      = mm->add_parameter("A", migraphx::shape{migraphx::shape::float_type, {2, 7}});
    auto b      = mm->add_parameter("B", migraphx::shape{migraphx::shape::uint8_type, {18}});
    auto absmax = mm->add_parameter("absmax", migraphx::shape{migraphx::shape::float_type, {3}});

    auto unpacked_b = mm->add_instruction(migraphx::make_op("unpack_int4"), b);
    unpacked_b = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {5, 7}}}), unpacked_b);

    auto expanded_absmax =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), absmax);
    expanded_absmax = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {3, 16}}}), expanded_absmax);
    expanded_absmax =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {48}}}), expanded_absmax);
    expanded_absmax = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {35}}}),
        expanded_absmax);
    expanded_absmax =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {5, 7}}}), expanded_absmax);

    auto float_data = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), unpacked_b);
    auto scale_factor =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {8.0f}});
    auto scale_factor_bc = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {5, 7}}}), scale_factor);
    auto scaled_data = mm->add_instruction(migraphx::make_op("div"), float_data, scale_factor_bc);
    auto dequantized = mm->add_instruction(migraphx::make_op("mul"), scaled_data, expanded_absmax);

    dequantized =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dequantized);

    mm->add_instruction(migraphx::make_op("dot"), a, dequantized);

    auto prog = optimize_onnx("matmulbnb4_fp4_non_aligned_test.onnx");

    p.sort();
    prog.sort();
    EXPECT(p == prog);
}

TEST_CASE(matmulbnb4_fp4_1d_input_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto a      = mm->add_parameter("A", migraphx::shape{migraphx::shape::float_type, {8}});
    auto b      = mm->add_parameter("B", migraphx::shape{migraphx::shape::uint8_type, {16}});
    auto absmax = mm->add_parameter("absmax", migraphx::shape{migraphx::shape::float_type, {2}});

    auto unpacked_b = mm->add_instruction(migraphx::make_op("unpack_int4"), b);
    unpacked_b = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 8}}}), unpacked_b);

    auto expanded_absmax =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), absmax);
    expanded_absmax = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {2, 16}}}), expanded_absmax);
    expanded_absmax =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {32}}}), expanded_absmax);
    expanded_absmax =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 8}}}), expanded_absmax);

    auto float_data = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), unpacked_b);
    auto scale_factor =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {8.0f}});
    auto scale_factor_bc = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {4, 8}}}), scale_factor);
    auto scaled_data = mm->add_instruction(migraphx::make_op("div"), float_data, scale_factor_bc);
    auto dequantized = mm->add_instruction(migraphx::make_op("mul"), scaled_data, expanded_absmax);

    dequantized =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dequantized);

    auto a_unsqueezed = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), a);
    auto dot = mm->add_instruction(migraphx::make_op("dot"), a_unsqueezed, dequantized);
    mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0}}}), dot);

    auto prog = optimize_onnx("matmulbnb4_fp4_1d_input_test.onnx");

    p.sort();
    prog.sort();
    EXPECT(p == prog);
}

TEST_CASE(matmulbnb4_fp4_3d_input_test)
{
    migraphx::program p;
    auto* mm    = p.get_main_module();
    auto a      = mm->add_parameter("A", migraphx::shape{migraphx::shape::float_type, {2, 3, 8}});
    auto b      = mm->add_parameter("B", migraphx::shape{migraphx::shape::uint8_type, {16}});
    auto absmax = mm->add_parameter("absmax", migraphx::shape{migraphx::shape::float_type, {2}});

    auto unpacked_b = mm->add_instruction(migraphx::make_op("unpack_int4"), b);
    unpacked_b = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 8}}}), unpacked_b);

    auto expanded_absmax =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), absmax);
    expanded_absmax = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {2, 16}}}), expanded_absmax);
    expanded_absmax =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {32}}}), expanded_absmax);
    expanded_absmax =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 8}}}), expanded_absmax);

    auto float_data = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), unpacked_b);
    auto scale_factor =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {8.0f}});
    auto scale_factor_bc = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {4, 8}}}), scale_factor);
    auto scaled_data = mm->add_instruction(migraphx::make_op("div"), float_data, scale_factor_bc);
    auto dequantized = mm->add_instruction(migraphx::make_op("mul"), scaled_data, expanded_absmax);

    dequantized =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {1, 0}}}), dequantized);

    auto dequantized_bc = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 8}}}), dequantized);

    mm->add_instruction(migraphx::make_op("dot"), a, dequantized_bc);

    auto prog = optimize_onnx("matmulbnb4_fp4_3d_input_test.onnx");

    p.sort();
    prog.sort();
    EXPECT(p == prog);
}

TEST_CASE(matmulbnb4_invalid_quant_type_test)
{
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_invalid_quant_type_test.onnx"); }));
}

TEST_CASE(matmulbnb4_invalid_block_size_test)
{
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_invalid_block_size_test.onnx"); }));
}

TEST_CASE(matmulbnb4_invalid_block_size_small_test)
{
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_invalid_block_size_small_test.onnx"); }));
}

TEST_CASE(matmulbnb4_wrong_input_count_test)
{
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_wrong_input_count_test.onnx"); }));
}

TEST_CASE(matmulbnb4_wrong_a_dims_test)
{
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_wrong_a_dims_test.onnx"); }));
}

TEST_CASE(matmulbnb4_wrong_a_inner_dim_test)
{
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_wrong_a_inner_dim_test.onnx"); }));
}

TEST_CASE(matmulbnb4_wrong_b_dims_test)
{
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_wrong_b_dims_test.onnx"); }));
}

TEST_CASE(matmulbnb4_wrong_absmax_dims_test)
{
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_wrong_absmax_dims_test.onnx"); }));
}

TEST_CASE(matmulbnb4_missing_n_attr_test)
{
    EXPECT(test::throws(
        [&] { migraphx::program p = read_onnx("matmulbnb4_missing_n_attr_test.onnx"); }));
}
