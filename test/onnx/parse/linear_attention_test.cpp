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

static migraphx::instruction_ref to_float(migraphx::module& m, migraphx::instruction_ref ins)
{
    return m.add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), ins);
}

// Single decode step (T=1) of the linear update rule with multi-query heads:
// S = past + k (x) v; o = scale * q^T S, accumulated in float
TEST_CASE(linear_attention_decode_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto query = mm->add_parameter("query", {migraphx::shape::half_type, {1, 1, 4}});
    auto key   = mm->add_parameter("key", {migraphx::shape::half_type, {1, 1, 2}});
    auto value = mm->add_parameter("value", {migraphx::shape::half_type, {1, 1, 2}});
    auto past  = mm->add_parameter("past_state", {migraphx::shape::half_type, {1, 1, 2, 2}});

    auto q = to_float(*mm, query);
    q      = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 1, 2, 2}}}), q);
    q = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1, 4}}}), q);
    auto k = to_float(*mm, key);
    k      = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 1, 2}}}), k);
    k = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), k);
    auto v = to_float(*mm, value);
    v      = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 1, 2}}}), v);
    v = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), v);
    auto state = to_float(*mm, past);

    auto k_col = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {1}}}), k);
    auto v_t = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {1}}}), v);
    auto outer = mm->add_instruction(migraphx::make_op("dot"), k_col, v_t);
    state      = mm->add_instruction(migraphx::make_op("add"), state, outer);
    auto q_t   = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {3}}, {"starts", {0}}, {"ends", {1}}}), q);
    q_t      = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), q_t);
    auto o_t = mm->add_instruction(migraphx::make_op("dot"), q_t, state);
    o_t      = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {3}}}), o_t);

    auto scale = mm->add_literal(migraphx::literal{
        migraphx::shape{migraphx::shape::float_type, {1}}, {1.0f / std::sqrt(2.0f)}});
    scale      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 1, 2, 1, 2}}}), scale);
    auto output = mm->add_instruction(migraphx::make_op("mul"), o_t, scale);
    output = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2, 4}}}),
                                 output);
    output = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 4}}}), output);
    mm->add_instruction(migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
                        output);
    mm->add_instruction(migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
                        state);

    auto prog = optimize_onnx("linear_attention_decode_test.onnx");
    EXPECT(p == prog);
}

// Multi-step linear rule: masked-matmul form O = scale * tril(Q K^T) V,
// present = K^T V
TEST_CASE(linear_attention_linear_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto query = mm->add_parameter("query", {migraphx::shape::float_type, {2, 3, 4}});
    auto key   = mm->add_parameter("key", {migraphx::shape::float_type, {2, 3, 4}});
    auto value = mm->add_parameter("value", {migraphx::shape::float_type, {2, 3, 4}});

    auto q = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 2, 1, 2}}}), query);
    q = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1, 4}}}), q);
    auto k = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 2, 2}}}), key);
    k = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), k);
    auto v = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 2, 2}}}), value);
    v = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), v);

    auto k5     = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), k);
    auto scores = mm->add_instruction(migraphx::make_op("dot"), q, k5);

    std::vector<int8_t> mask_data{0, 1, 1, 0, 0, 1, 0, 0, 0};
    auto mask = mm->add_literal(migraphx::literal{{migraphx::shape::bool_type, {3, 3}}, mask_data});
    mask = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 1, 3, 3}}}),
                               mask);
    auto zero = mm->add_literal(migraphx::literal{{migraphx::shape::float_type, {1}}, {0.0f}});
    zero = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 1, 3, 3}}}),
                               zero);
    scores = mm->add_instruction(migraphx::make_op("where"), mask, zero, scores);

    auto v5     = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), v);
    auto output = mm->add_instruction(migraphx::make_op("dot"), scores, v5);

    auto scale = mm->add_literal(migraphx::literal{
        migraphx::shape{migraphx::shape::float_type, {1}}, {1.0f / std::sqrt(2.0f)}});
    scale      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {2, 2, 1, 3, 2}}}), scale);
    output = mm->add_instruction(migraphx::make_op("mul"), output, scale);
    output = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2, 4}}}),
                                 output);
    output = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 4}}}), output);
    mm->add_instruction(migraphx::make_op("dot"), k, v);

    auto prog = optimize_onnx("linear_attention_linear_test.onnx");
    EXPECT(p == prog);
}

// fp16 GQA with past state: the matmuls run in fp16; the mask stage and the
// scale/add epilogue accumulate in float (a fused gemm-gemm kernel elides
// the float->half->float round trip between the two matmuls)
TEST_CASE(linear_attention_linear_fp16_test)
{
    migraphx::program p;
    auto* mm   = p.get_main_module();
    auto query = mm->add_parameter("query", {migraphx::shape::half_type, {1, 4, 8}});
    auto key   = mm->add_parameter("key", {migraphx::shape::half_type, {1, 4, 4}});
    auto value = mm->add_parameter("value", {migraphx::shape::half_type, {1, 4, 4}});
    auto past  = mm->add_parameter("past_state", {migraphx::shape::half_type, {1, 2, 2, 2}});

    auto q = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 2, 2, 2}}}), query);
    q = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1, 4}}}), q);
    auto k = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 2, 2}}}), key);
    k = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), k);
    auto v = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 2, 2}}}), value);
    v = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), v);

    auto k5 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), k);
    k5 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 2, 4}}}),
                             k5);
    auto scores = mm->add_instruction(migraphx::make_op("dot"), q, k5);
    scores      = to_float(*mm, scores);

    std::vector<int8_t> mask_data{0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0};
    auto mask = mm->add_literal(migraphx::literal{{migraphx::shape::bool_type, {4, 4}}, mask_data});
    mask = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 4, 4}}}),
                               mask);
    auto zero = mm->add_literal(migraphx::literal{{migraphx::shape::float_type, {1}}, {0.0f}});
    zero = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 4, 4}}}),
                               zero);
    scores = mm->add_instruction(migraphx::make_op("where"), mask, zero, scores);
    scores = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), scores);

    auto v5 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), v);
    v5 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 4, 2}}}),
                             v5);
    auto output = mm->add_instruction(migraphx::make_op("dot"), scores, v5);
    output      = to_float(*mm, output);

    auto s05 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), past);
    s05 = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 2, 2}}}),
                              s05);
    auto init = mm->add_instruction(migraphx::make_op("dot"), q, s05);
    init      = to_float(*mm, init);
    output    = mm->add_instruction(migraphx::make_op("add"), output, init);

    auto scale = mm->add_literal(migraphx::literal{
        migraphx::shape{migraphx::shape::float_type, {1}}, {1.0f / std::sqrt(2.0f)}});
    scale      = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 4, 2}}}), scale);
    output = mm->add_instruction(migraphx::make_op("mul"), output, scale);
    output = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2, 4}}}),
                                 output);
    output = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 8}}}), output);
    mm->add_instruction(migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
                        output);

    auto state   = mm->add_instruction(migraphx::make_op("dot"), k, v);
    auto state_f = to_float(*mm, state);
    auto s0      = to_float(*mm, past);
    auto sum     = mm->add_instruction(migraphx::make_op("add"), state_f, s0);
    mm->add_instruction(migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
                        sum);

    auto prog = optimize_onnx("linear_attention_linear_fp16_test.onnx");
    EXPECT(p == prog);
}

TEST_CASE(linear_attention_output_shapes_test)
{
    auto prog   = read_onnx("linear_attention_gated_delta_test.onnx");
    auto shapes = prog.get_output_shapes();
    EXPECT(shapes.size() == 2);
    EXPECT(shapes.front() == migraphx::shape{migraphx::shape::float_type, {1, 3, 8}});
    EXPECT(shapes.back() == migraphx::shape{migraphx::shape::float_type, {1, 2, 2, 2}});
}

TEST_CASE(linear_attention_bad_rule_test)
{
    EXPECT(test::throws([&] { read_onnx("linear_attention_bad_rule_test.onnx"); }));
}

TEST_CASE(linear_attention_missing_heads_test)
{
    EXPECT(test::throws([&] { read_onnx("linear_attention_missing_heads_test.onnx"); }));
}

TEST_CASE(linear_attention_bad_heads_test)
{
    EXPECT(test::throws([&] { read_onnx("linear_attention_bad_heads_test.onnx"); }));
}

TEST_CASE(linear_attention_missing_decay_test)
{
    EXPECT(test::throws([&] { read_onnx("linear_attention_missing_decay_test.onnx"); }));
}

TEST_CASE(linear_attention_forbidden_beta_test)
{
    EXPECT(test::throws([&] { read_onnx("linear_attention_forbidden_beta_test.onnx"); }));
}

TEST_CASE(linear_attention_bad_decay_test)
{
    EXPECT(test::throws([&] { read_onnx("linear_attention_bad_decay_test.onnx"); }));
}

TEST_CASE(linear_attention_bad_past_test)
{
    EXPECT(test::throws([&] { read_onnx("linear_attention_bad_past_test.onnx"); }));
}
