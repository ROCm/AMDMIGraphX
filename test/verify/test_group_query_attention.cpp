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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/float_equal.hpp>

// NOLINTNEXTLINE(readability-function-size)
static migraphx::program create_gqa_program(const size_t batch_size,
                                            const size_t num_heads,
                                            const size_t kv_num_heads,
                                            const size_t sequence_length,
                                            const size_t head_size,
                                            const size_t past_sequence_length,
                                            const size_t max_sequence_length,
                                            const bool do_rotary,
                                            const float scale,
                                            const bool test_rotary      = false,
                                            const bool test_concat      = false,
                                            const int local_window_size = -1)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    std::vector<size_t> query_lens{
        batch_size, sequence_length, head_size * (num_heads + 2 * kv_num_heads)};
    std::vector<size_t> kv_lens{batch_size, kv_num_heads, max_sequence_length, head_size};
    std::vector<size_t> slk_lens{batch_size, 1};
    std::vector<size_t> cs_cache_lens{max_sequence_length, head_size / 2};
    auto dtype = migraphx::shape::half_type;
    migraphx::shape query_s{dtype, query_lens};
    migraphx::shape kv_s{dtype, kv_lens};
    migraphx::shape slk_s{migraphx::shape::int32_type, slk_lens};
    migraphx::shape cs_cache_s{dtype, cs_cache_lens};
    auto query = mm->add_parameter("query", query_s);
    std::vector<int> slk_vec(slk_s.elements(), past_sequence_length);
    std::vector<float> cs_min_vec(cs_cache_s.elements(), -1.0);
    std::vector<float> cs_max_vec(cs_cache_s.elements(), 1.0);
    auto k         = mm->add_parameter("k", kv_s);
    auto v         = mm->add_parameter("v", kv_s);
    auto slk       = mm->add_parameter("slk", slk_s);
    auto slk_lit   = mm->add_literal(slk_s, slk_vec);
    slk            = mm->add_instruction(migraphx::make_op("clip"), slk, slk_lit, slk_lit);
    auto cs_min    = mm->add_literal(cs_cache_s, cs_min_vec);
    auto cs_max    = mm->add_literal(cs_cache_s, cs_max_vec);
    auto cos_cache = mm->add_parameter("cos_cache", cs_cache_s);
    auto sin_cache = mm->add_parameter("sin_cache", cs_cache_s);
    cos_cache      = mm->add_instruction(migraphx::make_op("clip"), cos_cache, cs_min, cs_max);
    sin_cache      = mm->add_instruction(migraphx::make_op("clip"), sin_cache, cs_min, cs_max);

    std::vector<std::size_t> bsnh{
        batch_size, sequence_length, num_heads + 2 * kv_num_heads, head_size};

    auto transposed_qkv =
        mm->add_instruction(migraphx::make_op("reshape", {{"dims", bsnh}}), query);

    transposed_qkv = mm->add_instruction(
        migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), transposed_qkv);

    auto rotary_qkv = transposed_qkv;
    if(do_rotary)
    {
        std::vector<migraphx::instruction_ref> rotary_inputs{
            transposed_qkv, slk, cos_cache, sin_cache};
        rotary_qkv = mm->add_instruction(
            migraphx::make_op(
                "gqa_rotary_embedding",
                {{"kv_num_heads", kv_num_heads}, {"num_heads", num_heads}, {"interleaved", false}}),
            rotary_inputs);
        if(test_rotary)
        {
            mm->add_return({rotary_qkv});
            return p;
        }
    }

    auto rotary_k = mm->add_instruction(
        migraphx::make_op(
            "slice",
            {{"axes", {1}}, {"starts", {num_heads}}, {"ends", {num_heads + kv_num_heads}}}),
        rotary_qkv);
    auto rotary_v =
        mm->add_instruction(migraphx::make_op("slice",
                                              {{"axes", {1}},
                                               {"starts", {num_heads + kv_num_heads}},
                                               {"ends", {num_heads + (2 * kv_num_heads)}}}),
                            rotary_qkv);
    std::vector<migraphx::instruction_ref> concat_k_inputs{rotary_k, slk, k};
    std::vector<migraphx::instruction_ref> concat_v_inputs{rotary_v, slk, v};

    k = mm->add_instruction(
        migraphx::make_op("concat_past_present", {{"kv_num_heads", kv_num_heads}}),
        concat_k_inputs);
    v = mm->add_instruction(
        migraphx::make_op("concat_past_present", {{"kv_num_heads", kv_num_heads}}),
        concat_v_inputs);

    auto k_out = k;
    auto v_out = v;

    if(test_concat)
    {
        mm->add_return({k_out, v_out});
        return p;
    }

    auto kv_num_heads_factor = num_heads / kv_num_heads;
    auto max_seq_len         = kv_s.lens()[2];
    auto past_sl             = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", {batch_size, num_heads}}}), slk);

    auto q = mm->add_instruction(
        migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {num_heads}}}),
        rotary_qkv);

    if(kv_num_heads_factor != 1)
    {
        auto kv_new_lens  = kv_lens;
        kv_new_lens.at(1) = num_heads;
        k                 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), k);
        v                 = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), v);
        auto kv_unsqueezed_lens = kv_lens;
        kv_unsqueezed_lens.insert(kv_unsqueezed_lens.begin() + 2, kv_num_heads_factor);
        k = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", kv_unsqueezed_lens}}), k);
        v = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", kv_unsqueezed_lens}}), v);
        k = mm->add_instruction(migraphx::make_op("reshape", {{"dims", kv_new_lens}}), k);
        v = mm->add_instruction(migraphx::make_op("reshape", {{"dims", kv_new_lens}}), v);
    }
    auto kt =
        mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);
    auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), q, kt);

    std::vector<int> range_vec(max_seq_len);
    std::iota(range_vec.begin(), range_vec.end(), 0);
    migraphx::shape range_s{slk_s.type(), {max_seq_len}};
    auto range = mm->add_literal(range_s, range_vec);
    std::vector<std::size_t> bnsm{batch_size, num_heads, sequence_length, max_seq_len};
    auto bc_range =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", bnsm}}), range);

    auto scalar_s = migraphx::shape{query_s.type(), {1}};
    auto ninf =
        mm->add_literal(migraphx::literal{scalar_s, {-std::numeric_limits<float>::infinity()}});
    ninf = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", bnsm}}), ninf);

    auto scale_ins = mm->add_literal(migraphx::literal{scalar_s, {scale}});
    scale_ins =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", bnsm}}), scale_ins);
    auto mul = mm->add_instruction(migraphx::make_op("mul"), gemm1, scale_ins);

    migraphx::instruction_ref seq_range;
    if(sequence_length > 1)
    {
        std::vector<int> seq_range_vec(sequence_length);
        std::iota(seq_range_vec.begin(), seq_range_vec.end(), 0);
        migraphx::shape seq_range_s{slk_s.type(), {sequence_length}};
        seq_range = mm->add_literal(seq_range_s, seq_range_vec);
        seq_range = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {sequence_length, 1}}}), seq_range);
        seq_range = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", bnsm}}),
                                        seq_range);
        auto causal_mask = mm->add_instruction(migraphx::make_op("greater"), bc_range, seq_range);
        causal_mask      = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}),
            causal_mask);
        mul = mm->add_instruction(migraphx::make_op("where"), causal_mask, ninf, mul);
    }

    auto bc_past_sl = mm->add_instruction(
        migraphx::make_op("reshape", {{"dims", {batch_size, num_heads, 1, 1}}}), past_sl);
    auto mask_comp =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", bnsm}}), bc_past_sl);

    if(local_window_size > 0)
    {
        bool is_prompt       = sequence_length > 1;
        auto window_size_lit = mm->add_literal(
            migraphx::literal{migraphx::shape{slk_s.type(), {1}},
                              {is_prompt ? -local_window_size : -(local_window_size + 1)}});
        window_size_lit = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", bnsm}}), window_size_lit);
        auto window_comp = mm->add_instruction(
            migraphx::make_op("add"), is_prompt ? seq_range : mask_comp, window_size_lit);
        auto window_mask = mm->add_instruction(migraphx::make_op("greater"), window_comp, bc_range);
        window_mask      = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}),
            window_mask);
        mul = mm->add_instruction(migraphx::make_op("where"), window_mask, ninf, mul);
    }

    auto mask = mm->add_instruction(migraphx::make_op("greater"), bc_range, mask_comp);
    mask      = mm->add_instruction(
        migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), mask);
    auto where   = mm->add_instruction(migraphx::make_op("where"), mask, ninf, mul);
    auto softmax = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), where);
    auto scores  = mm->add_instruction(migraphx::make_op("dot"), softmax, v);
    auto out = mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}),
                                   scores);
    out      = mm->add_instruction(
        migraphx::make_op("reshape",
                               {{"dims", {batch_size, sequence_length, head_size * num_heads}}}),
        out);

    mm->add_return({out, k_out, v_out});
    return p;
}

struct test_group_query_attention_decode_small
    : verify_program<test_group_query_attention_decode_small>
{
    migraphx::program create_program() const
    {
        return create_gqa_program(/* batch_size=           */ 2,
                                  /* num_heads=            */ 2,
                                  /* kv_num_heads=         */ 2,
                                  /* sequence_length=      */ 1,
                                  /* head_size=            */ 2,
                                  /* past_sequence_length= */ 3,
                                  /* max_sequence_length=  */ 4,
                                  /* do_rotary=            */ true,
                                  /* scale=                */ 0.5);
    }
    std::string section() const { return "attention"; }
};

struct test_group_query_attention_decode : verify_program<test_group_query_attention_decode>
{
    migraphx::program create_program() const
    {
        return create_gqa_program(/* batch_size=           */ 1,
                                  /* num_heads=            */ 32,
                                  /* kv_num_heads=         */ 32,
                                  /* sequence_length=      */ 1,
                                  /* head_size=            */ 128,
                                  /* past_sequence_length= */ 15,
                                  /* max_sequence_length=  */ 2048,
                                  /* do_rotary=            */ true,
                                  /* scale=                */ 1.0 / sqrt(128.0));
    }
    std::string section() const { return "attention"; }
};

struct test_group_query_attention_prefill_small
    : verify_program<test_group_query_attention_prefill_small>
{
    migraphx::program create_program() const
    {
        return create_gqa_program(/* batch_size=           */ 1,
                                  /* num_heads=            */ 2,
                                  /* kv_num_heads=         */ 2,
                                  /* sequence_length=      */ 2,
                                  /* head_size=            */ 2,
                                  /* past_sequence_length= */ 2,
                                  /* max_sequence_length=  */ 4,
                                  /* do_rotary=            */ true,
                                  /* scale=                */ 0.5);
    }
    std::string section() const { return "attention"; }
};

struct test_group_query_attention_prefill : verify_program<test_group_query_attention_prefill>
{
    migraphx::program create_program() const
    {
        return create_gqa_program(/* batch_size=           */ 1,
                                  /* num_heads=            */ 32,
                                  /* kv_num_heads=         */ 32,
                                  /* sequence_length=      */ 5,
                                  /* head_size=            */ 128,
                                  /* past_sequence_length= */ 5,
                                  /* max_sequence_length=  */ 2048,
                                  /* do_rotary=            */ true,
                                  /* scale=                */ 1.0);
    }
    std::string section() const { return "attention"; }
};

struct test_group_query_attention_no_rotary : verify_program<test_group_query_attention_no_rotary>
{
    migraphx::program create_program() const
    {
        return create_gqa_program(/* batch_size=           */ 1,
                                  /* num_heads=            */ 32,
                                  /* kv_num_heads=         */ 32,
                                  /* sequence_length=      */ 5,
                                  /* head_size=            */ 128,
                                  /* past_sequence_length= */ 5,
                                  /* max_sequence_length=  */ 1024,
                                  /* do_rotary=            */ false,
                                  /* scale=                */ 1.0 / sqrt(128.0));
    }
    std::string section() const { return "attention"; }
};

struct test_group_query_attention_grouped : verify_program<test_group_query_attention_grouped>
{
    migraphx::program create_program() const
    {
        return create_gqa_program(/* batch_size=           */ 1,
                                  /* num_heads=            */ 32,
                                  /* kv_num_heads=         */ 8,
                                  /* sequence_length=      */ 1,
                                  /* head_size=            */ 128,
                                  /* past_sequence_length= */ 15,
                                  /* max_sequence_length=  */ 2048,
                                  /* do_rotary=            */ true,
                                  /* scale=                */ 1.0 / sqrt(128.0));
    }
    //std::string section() const { return "attention"; }
};

struct test_group_query_attention_rotary_only
    : verify_program<test_group_query_attention_rotary_only>
{
    migraphx::program create_program() const
    {
        return create_gqa_program(/* batch_size=           */ 1,
                                  /* num_heads=            */ 32,
                                  /* kv_num_heads=         */ 32,
                                  /* sequence_length=      */ 1,
                                  /* head_size=            */ 128,
                                  /* past_sequence_length= */ 15,
                                  /* max_sequence_length=  */ 2048,
                                  /* do_rotary=            */ true,
                                  /* scale=                */ 1.0 / sqrt(128.0),
                                  /* test_rotary=          */ true,
                                  /* test_concat=          */ false);
    }
    std::string section() const { return "attention"; }
};

struct test_group_query_attention_concat_only
    : verify_program<test_group_query_attention_concat_only>
{
    migraphx::program create_program() const
    {
        return create_gqa_program(/* batch_size=           */ 2,
                                  /* num_heads=            */ 32,
                                  /* kv_num_heads=         */ 32,
                                  /* sequence_length=      */ 1,
                                  /* head_size=            */ 128,
                                  /* past_sequence_length= */ 15,
                                  /* max_sequence_length=  */ 2048,
                                  /* do_rotary=            */ true,
                                  /* scale=                */ 1.0 / sqrt(128.0),
                                  /* test_rotary=          */ false,
                                  /* test_concat=          */ true);
    }
    std::string section() const { return "attention"; }
};

struct test_group_query_attention_concat_only_small
    : verify_program<test_group_query_attention_concat_only_small>
{
    migraphx::program create_program() const
    {
        return create_gqa_program(/* batch_size=           */ 1,
                                  /* num_heads=            */ 14,
                                  /* kv_num_heads=         */ 2,
                                  /* sequence_length=      */ 1,
                                  /* head_size=            */ 8,
                                  /* past_sequence_length= */ 4,
                                  /* max_sequence_length=  */ 8,
                                  /* do_rotary=            */ true,
                                  /* scale=                */ 1.0 / sqrt(8.0),
                                  /* test_rotary=          */ false,
                                  /* test_concat=          */ true);
    }
    std::string section() const { return "attention"; }
};

struct test_group_query_attention_prefill_local
    : verify_program<test_group_query_attention_prefill_local>
{
    migraphx::program create_program() const
    {
        return create_gqa_program(/* batch_size=           */ 1,
                                  /* num_heads=            */ 2,
                                  /* kv_num_heads=         */ 2,
                                  /* sequence_length=      */ 4,
                                  /* head_size=            */ 2,
                                  /* past_sequence_length= */ 4,
                                  /* max_sequence_length=  */ 6,
                                  /* do_rotary=            */ true,
                                  /* scale=                */ 0.5,
                                  /* test_rotary=          */ false,
                                  /* test_concat=          */ false,
                                  /* local_window_size=    */ 2);
    }
    //std::string section() const { return "attention"; }
};

struct test_group_query_attention_decode_local
    : verify_program<test_group_query_attention_decode_local>
{
    migraphx::program create_program() const
    {
        return create_gqa_program(/* batch_size=           */ 1,
                                  /* num_heads=            */ 2,
                                  /* kv_num_heads=         */ 2,
                                  /* sequence_length=      */ 1,
                                  /* head_size=            */ 2,
                                  /* past_sequence_length= */ 4,
                                  /* max_sequence_length=  */ 8,
                                  /* do_rotary=            */ true,
                                  /* scale=                */ 0.5,
                                  /* test_rotary=          */ false,
                                  /* test_concat=          */ false,
                                  /* local_window_size=    */ 2);
    }
    //std::string section() const { return "attention"; }
};
