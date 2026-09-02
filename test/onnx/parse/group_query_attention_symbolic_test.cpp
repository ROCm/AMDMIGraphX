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
#include <onnx_test_utils.hpp>

#include <migraphx/common.hpp>

// GroupQueryAttention with a symbolic sequence length: every parse-time prompt/decode
// branch of the static parse is replaced with the unified run-time form. Per-token
// positions of the current tokens are seqlens_k + 1 - seq + i and the causal mask is
// j > position, which covers both the prompt (seqlens_k = seq - 1) and decode (seq = 1)
// cases.
TEST_CASE(group_query_attention_symbolic_test)
{
    using migraphx::sym::lit;
    using migraphx::sym::var;

    const auto seq = var("sequence_length", {1, 8});
    const migraphx::shape qkv_s{migraphx::shape::half_type,
                                sym_dims({lit(int64_t{1}), seq, lit(int64_t{128})})};

    migraphx::program p;
    auto* mm = p.get_main_module();

    // Initializers in graph order: seqlens_k, total_sequence_length, cos_cache, sin_cache
    const migraphx::shape slk_s{migraphx::shape::int32_type, {1, 1}};
    const migraphx::shape cache_s{migraphx::shape::half_type, {10, 8}};
    auto slk = mm->add_literal(migraphx::literal{slk_s, {1}});
    mm->add_literal(migraphx::literal{slk_s, {10}});
    std::vector<float> cache_vals(cache_s.elements(), 1.0f);
    auto cos_cache = mm->add_literal(migraphx::literal{cache_s, cache_vals});
    auto sin_cache = mm->add_literal(migraphx::literal{cache_s, cache_vals});

    // Parameters in graph input order
    const migraphx::shape key_value_s{migraphx::shape::float_type, {1}};
    const migraphx::shape kv_s{migraphx::shape::half_type, {1, 2, 10, 16}};
    auto qkv = mm->add_parameter("qkv", qkv_s);
    mm->add_parameter("key", key_value_s);
    mm->add_parameter("value", key_value_s);
    auto past_k = mm->add_parameter("past_key_values_key", kv_s);
    auto past_v = mm->add_parameter("past_key_values_value", kv_s);

    // num_heads=4, kv_num_heads=2, head_size=16, max_seq_len=10, scale=0.25
    auto transposed_qkv =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", {0, -1, 8, 16}}}), qkv);
    transposed_qkv = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), transposed_qkv);
    auto qk = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {6}}}),
        transposed_qkv);
    auto cur_v = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {6}}, {"ends", {8}}}),
        transposed_qkv);

    // Per-token positions of the current tokens {batch, seq, 1}: seqlens_k + 1 - seq + i
    auto slk64 = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::int64_type}}), slk);
    slk64 = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {-1}}}), slk64);
    auto seq_rt =
        mm->add_instruction(migraphx::make_op("dimensions_of", {{"start", 1}, {"end", 2}}), qkv);
    auto zero =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {0}});
    auto one =
        mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1}}, {1}});
    auto past_len = migraphx::add_common_op(*mm, migraphx::make_op("sub"), {slk64, seq_rt});
    past_len      = migraphx::add_common_op(*mm, migraphx::make_op("add"), {past_len, one});
    auto iota     = mm->add_instruction(
        migraphx::make_op(
            "dynamic_range",
            {{"output_dim", migraphx::to_value(migraphx::shape::dynamic_dimension{seq})}}),
        zero,
        seq_rt,
        one);
    auto past_len_b =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1, 2}}}), past_len);
    auto iota_b    = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 2}}}), iota);
    auto positions = migraphx::add_common_op(*mm, migraphx::make_op("add"), {past_len_b, iota_b});

    qk = migraphx::op::builder::add("rotary_embedding",
                                    *mm,
                                    {qk, positions, cos_cache, sin_cache},
                                    {{"interleaved", false}})
             .at(0);

    auto q = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {4}}}), qk);
    auto cur_k = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {4}}, {"ends", {6}}}), qk);

    auto k = mm->add_instruction(
        migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}), cur_k, slk, past_k);
    auto v = mm->add_instruction(
        migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}), cur_v, slk, past_v);
    auto k_out = k;
    auto v_out = v;

    // Repeat the kv heads for the grouped queries (factor 2)
    k = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), k);
    v = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), v);
    k = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 10, 16}}}),
                            k);
    v = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 10, 16}}}),
                            v);
    k = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 10, 16}}}), k);
    v = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 10, 16}}}), v);
    auto kt =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);

    const auto& q_dims = q->get_shape().dyn_dims();
    auto bcast_for_dot = [&](migraphx::instruction_ref static_ins) {
        auto dims = static_ins->get_shape().to_symbolic().dyn_dims();
        dims[0]   = q_dims[0];
        return mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_dyn_dims", migraphx::to_value(dims)}}),
            static_ins);
    };
    auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), q, bcast_for_dot(kt));

    auto scale_lit = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::half_type, {1}}, {0.25f}});
    auto scaled = migraphx::add_common_op(*mm, migraphx::make_op("mul"), {gemm1, scale_lit});

    // Causal mask: cache column j is masked when j > position of the query token
    std::vector<int64_t> col_vec(10);
    std::iota(col_vec.begin(), col_vec.end(), 0);
    auto col = mm->add_literal(
        migraphx::literal{migraphx::shape{migraphx::shape::int64_type, {1, 1, 1, 10}}, col_vec});
    auto positions_b =
        mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), positions);
    auto mask = migraphx::add_common_op(*mm, migraphx::make_op("greater"), {col, positions_b});
    mask      = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), mask);

    const auto out_dims = migraphx::to_value(scaled->get_shape().dyn_dims());
    mask = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_dyn_dims", out_dims}}),
                               mask);
    auto ninf = mm->add_literal(migraphx::literal{migraphx::shape{migraphx::shape::half_type, {1}},
                                                  {-std::numeric_limits<float>::infinity()}});
    ninf = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_dyn_dims", out_dims}}),
                               ninf);
    auto where   = mm->add_instruction(migraphx::make_op("where"), mask, ninf, scaled);
    auto softmax = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), where);
    auto scores  = mm->add_instruction(migraphx::make_op("dot"), softmax, bcast_for_dot(v));
    auto out = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}),
                                   scores);
    out      = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {0, -1, 64}}}), out);
    mm->add_return({out, k_out, v_out});

    migraphx::onnx_options options;
    options.map_dyn_input_dims["qkv"] = qkv_s.dyn_dims();
    options.use_symbolic_shapes       = true;
    auto prog = read_onnx("group_query_attention_grouped_test.onnx", options);

    EXPECT(p == prog);
}
