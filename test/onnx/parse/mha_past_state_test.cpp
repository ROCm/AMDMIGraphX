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

TEST_CASE(multi_head_attention_past_state_test)
{
    const int64_t batch_size            = 1;
    const int64_t sequence_length       = 1;
    const int64_t past_sequence_length  = 3;
    const int64_t total_sequence_length = sequence_length + past_sequence_length;
    const int64_t hidden_size           = 4;
    const int64_t num_heads             = 2;
    const int64_t head_size             = 2;
    const std::vector<std::size_t> input_lens{batch_size, sequence_length, hidden_size};
    const std::vector<std::size_t> reshape_lens{batch_size, sequence_length, num_heads, head_size};
    const std::vector<std::size_t> past_lens{batch_size, num_heads, past_sequence_length, head_size};
    const std::vector<std::size_t> concat_lens{batch_size, num_heads, total_sequence_length, head_size};
    const std::vector<int64_t> permutation{0, 2, 1, 3};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto q   = mm->add_parameter("q", {migraphx::shape::float_type, input_lens});
    auto k   = mm->add_parameter("k", {migraphx::shape::float_type, input_lens});
    auto v   = mm->add_parameter("v", {migraphx::shape::float_type, input_lens});
    auto past_k = mm->add_parameter("past_key", {migraphx::shape::float_type, past_lens});
    auto past_v = mm->add_parameter("past_value", {migraphx::shape::float_type, past_lens});
    auto seqlens_k = mm->add_parameter("past_sequence_length", {migraphx::shape::int32_type, {batch_size}});

    q = mm->add_instruction(migraphx::make_op("reshape", {{"dims", reshape_lens}}), q);
    k = mm->add_instruction(migraphx::make_op("reshape", {{"dims", reshape_lens}}), k);
    v = mm->add_instruction(migraphx::make_op("reshape", {{"dims", reshape_lens}}), v);

    q = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", permutation}}), q);
    k = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", permutation}}), k);
    v = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", permutation}}), v);

    // Concatenate with past states using concat_past_present
    k = mm->add_instruction(migraphx::make_op("concat_past_present", {{"kv_num_heads", num_heads}}),
                            k, seqlens_k, past_k);
    v = mm->add_instruction(migraphx::make_op("concat_past_present", {{"kv_num_heads", num_heads}}),
                            v, seqlens_k, past_v);

    const float scale = 1 / std::sqrt(head_size);
    auto scale_literal =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {scale}});

    auto key_transposed =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);

    auto result   = mm->add_instruction(migraphx::make_op("dot"), q, key_transposed);
    scale_literal = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", result->get_shape().lens()}}),
        scale_literal);
    result = mm->add_instruction(migraphx::make_op("dot"), result, scale_literal);
    result = mm->add_instruction(migraphx::make_op("softmax", {{"axis", -1}}), result);
    result = mm->add_instruction(migraphx::make_op("dot"), result, v);
    result =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", permutation}}), result);
    result = mm->add_instruction(migraphx::make_op("reshape", {{"dims", input_lens}}), result);

    auto prog = migraphx::parse_onnx("mha_past_state_test.onnx");

    EXPECT(p.sort() == prog.sort());
}

TEST_CASE(multi_head_attention_empty_past_state_test)
{
    // Test that empty past states are handled correctly
    const int64_t batch_size      = 1;
    const int64_t sequence_length = 2;
    const int64_t hidden_size     = 4;
    const int64_t num_heads       = 2;
    const int64_t head_size       = 2;
    const std::vector<std::size_t> input_lens{batch_size, sequence_length, hidden_size};
    const std::vector<std::size_t> reshape_lens{batch_size, sequence_length, num_heads, head_size};
    const std::vector<int64_t> permutation{0, 2, 1, 3};

    migraphx::program p;
    auto* mm = p.get_main_module();
    auto q   = mm->add_parameter("q", {migraphx::shape::float_type, input_lens});
    auto k   = mm->add_parameter("k", {migraphx::shape::float_type, input_lens});
    auto v   = mm->add_parameter("v", {migraphx::shape::float_type, input_lens});

    q = mm->add_instruction(migraphx::make_op("reshape", {{"dims", reshape_lens}}), q);
    k = mm->add_instruction(migraphx::make_op("reshape", {{"dims", reshape_lens}}), k);
    v = mm->add_instruction(migraphx::make_op("reshape", {{"dims", reshape_lens}}), v);

    q = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", permutation}}), q);
    k = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", permutation}}), k);
    v = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", permutation}}), v);

    const float scale = 1 / std::sqrt(head_size);
    auto scale_literal =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {scale}});

    auto key_transposed =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);

    auto result   = mm->add_instruction(migraphx::make_op("dot"), q, key_transposed);
    scale_literal = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", result->get_shape().lens()}}),
        scale_literal);
    result = mm->add_instruction(migraphx::make_op("mul"), result, scale_literal);
    result = mm->add_instruction(migraphx::make_op("softmax", {{"axis", -1}}), result);
    result = mm->add_instruction(migraphx::make_op("dot"), result, v);
    result =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", permutation}}), result);
    result = mm->add_instruction(migraphx::make_op("reshape", {{"dims", input_lens}}), result);

    auto prog = migraphx::parse_onnx("mha_empty_past_state_test.onnx");

    EXPECT(p.sort() == prog.sort());
}
