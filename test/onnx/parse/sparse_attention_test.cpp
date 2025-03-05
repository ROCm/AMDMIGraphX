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

TEST_CASE(sparse_attention_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto qkv = mm->add_parameter("qkv", migraphx::shape{migraphx::shape::float_type, {2, 32, 128}});
    auto key = mm->add_instruction(migraphx::make_op("undefined"));
    auto value = key;
    auto past_key =
        mm->add_parameter("past_key", migraphx::shape{migraphx::shape::float_type, {2, 2, 32, 16}});
    auto past_value = mm->add_parameter(
        "past_value", migraphx::shape{migraphx::shape::float_type, {2, 2, 32, 16}});
    auto block_row_indices = mm->add_parameter(
        "block_row_indices", migraphx::shape{migraphx::shape::int32_type, {2, 5}});
    auto block_col_indices = mm->add_parameter(
        "block_col_indices", migraphx::shape{migraphx::shape::int32_type, {2, 10}});
    auto total_sequence_length =
        mm->add_parameter("total_sequence_length", {migraphx::shape::int32_type, {1}});
    auto key_total_sequence_lengths =
        mm->add_parameter("key_total_sequence_lengths", {migraphx::shape::int32_type, {2}});

    qkv = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 32, 8, 16}}}), qkv);
    qkv = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), qkv);
    auto attn = mm->add_instruction(
        migraphx::make_op("sparse_attention",
                          {{"num_heads", 4}, {"kv_num_heads", 2}, {"sparse_block_size", 8}}),
        qkv,
        key,
        value,
        past_key,
        past_value,
        block_row_indices,
        block_col_indices,
        total_sequence_length,
        key_total_sequence_lengths);
    mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), attn);
    mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), attn);
    mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), attn);

    auto prog = optimize_onnx("sparse_attention_test.onnx");

    EXPECT(p == prog);
}

TEST_CASE(sparse_attention_rotary_test)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    auto qkv = mm->add_parameter("qkv", migraphx::shape{migraphx::shape::float_type, {2, 32, 128}});
    auto key = mm->add_instruction(migraphx::make_op("undefined"));
    auto value = key;
    auto past_key =
        mm->add_parameter("past_key", migraphx::shape{migraphx::shape::float_type, {2, 2, 32, 16}});
    auto past_value = mm->add_parameter(
        "past_value", migraphx::shape{migraphx::shape::float_type, {2, 2, 32, 16}});
    auto block_row_indices = mm->add_parameter(
        "block_row_indices", migraphx::shape{migraphx::shape::int32_type, {2, 5}});
    auto block_col_indices = mm->add_parameter(
        "block_col_indices", migraphx::shape{migraphx::shape::int32_type, {2, 10}});
    auto total_sequence_length =
        mm->add_parameter("total_sequence_length", {migraphx::shape::int32_type, {1}});
    auto key_total_sequence_lengths =
        mm->add_parameter("key_total_sequence_lengths", {migraphx::shape::int32_type, {2}});
    auto cos_cache =
        mm->add_parameter("cos_cache", migraphx::shape{migraphx::shape::float_type, {32, 8}});
    auto sin_cache =
        mm->add_parameter("sin_cache", migraphx::shape{migraphx::shape::float_type, {32, 8}});

    qkv = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 32, 8, 16}}}), qkv);
    qkv = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), qkv);
    auto attn = mm->add_instruction(migraphx::make_op("sparse_attention",
                                                      {{"num_heads", 4},
                                                       {"kv_num_heads", 2},
                                                       {"sparse_block_size", 8},
                                                       {"scale", 1.0f},
                                                       {"do_rotary", true},
                                                       {"rotary_interleaved", false}}),
                                    qkv,
                                    key,
                                    value,
                                    past_key,
                                    past_value,
                                    block_row_indices,
                                    block_col_indices,
                                    total_sequence_length,
                                    key_total_sequence_lengths,
                                    cos_cache,
                                    sin_cache);
    mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), attn);
    mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), attn);
    mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), attn);

    auto prog = optimize_onnx("sparse_attention_rotary_test.onnx");

    EXPECT(p == prog);
}

#define SPARSE_ATTENTION_NEGATIVE_TEST(name, msg)                                  \
    TEST_CASE(sparse_attention_##name##_test)                                      \
    {                                                                              \
        EXPECT(test::throws<migraphx::exception>(                                  \
            [&] { optimize_onnx("sparse_attention_" #name "_test.onnx"); }, msg)); \
    }

SPARSE_ATTENTION_NEGATIVE_TEST(missing_num_heads, "num_heads attribute is required")
SPARSE_ATTENTION_NEGATIVE_TEST(invalid_num_heads, "num_heads")

SPARSE_ATTENTION_NEGATIVE_TEST(missing_kv_num_heads, "kv_num_heads attribute is required")
SPARSE_ATTENTION_NEGATIVE_TEST(invalid_kv_num_heads, "kv_num_heads")

SPARSE_ATTENTION_NEGATIVE_TEST(missing_sparse_block_size, "sparse_block_size attribute is required")
SPARSE_ATTENTION_NEGATIVE_TEST(invalid_sparse_block_size, "sparse_block_size")

SPARSE_ATTENTION_NEGATIVE_TEST(too_few_inputs, "number of inputs")
SPARSE_ATTENTION_NEGATIVE_TEST(too_many_inputs, "number of inputs")

SPARSE_ATTENTION_NEGATIVE_TEST(qkv_invalid_rank, "query input rank")
SPARSE_ATTENTION_NEGATIVE_TEST(qkv_invalid_hidden_size, "hidden size")
SPARSE_ATTENTION_NEGATIVE_TEST(invalid_head_size, "head_size")
SPARSE_ATTENTION_NEGATIVE_TEST(rotary_invalid_head_size, "head_size")

SPARSE_ATTENTION_NEGATIVE_TEST(past_k_invalid_rank, "past_key rank")
SPARSE_ATTENTION_NEGATIVE_TEST(past_k_invalid_dim0, "past_key input dim 0")
SPARSE_ATTENTION_NEGATIVE_TEST(past_k_invalid_dim1, "past_key input dim 1")
SPARSE_ATTENTION_NEGATIVE_TEST(past_k_invalid_dim3, "past_key input dim 3")
SPARSE_ATTENTION_NEGATIVE_TEST(past_v_invalid_shape, "past_key and past_value")

SPARSE_ATTENTION_NEGATIVE_TEST(block_row_indices_invalid_rank, "block_row_indices input rank")
SPARSE_ATTENTION_NEGATIVE_TEST(block_row_indices_invalid_dim0, "block_row_indices input dim 0")
SPARSE_ATTENTION_NEGATIVE_TEST(block_row_indices_invalid_dim1, "block_row_indices input dim 1")
SPARSE_ATTENTION_NEGATIVE_TEST(block_col_indices_invalid_rank, "block_col_indices input rank")
SPARSE_ATTENTION_NEGATIVE_TEST(block_col_indices_invalid_dim0, "block_col_indices input dim 0")
SPARSE_ATTENTION_NEGATIVE_TEST(block_col_indices_invalid_dim1, "block_col_indices input dim 1")

SPARSE_ATTENTION_NEGATIVE_TEST(total_sequence_length_invalid_rank, "total_sequence_length input")
SPARSE_ATTENTION_NEGATIVE_TEST(total_sequence_length_invalid_len, "total_sequence_length input")

SPARSE_ATTENTION_NEGATIVE_TEST(key_total_sequence_lengths_invalid_rank,
                               "key_total_sequence_lengths input")
SPARSE_ATTENTION_NEGATIVE_TEST(key_total_sequence_lengths_invalid_len,
                               "key_total_sequence_lengths input")

SPARSE_ATTENTION_NEGATIVE_TEST(cos_cache_invalid_dim1, "cos_cache input dim 1")
SPARSE_ATTENTION_NEGATIVE_TEST(sin_cache_invalid_shape, "cos_cache and sin_cache")
