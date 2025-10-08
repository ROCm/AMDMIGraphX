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

struct test_group_query_attention_decode_small
    : verify_program<test_group_query_attention_decode_small>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();
        std::vector<size_t> query_lens{1, 1, 12};
        std::vector<size_t> kv_lens{1, 2, 4, 2};
        std::vector<size_t> slk_lens{1, 1};
        std::vector<size_t> tsl_lens{1, 1};
        std::vector<size_t> cs_cache_lens{4, 1};
        auto dtype = migraphx::shape::half_type;
        migraphx::shape query_s{dtype, query_lens};
        migraphx::shape kv_s{dtype, kv_lens};
        migraphx::shape slk_s{migraphx::shape::int32_type, slk_lens};
        migraphx::shape tsl_s{migraphx::shape::int32_type, tsl_lens};
        migraphx::shape cs_cache_s{dtype, cs_cache_lens};
        auto query = mm->add_parameter("query", query_s);
        std::vector<int> slk_vec(slk_s.elements(), 3);
        std::vector<int> tsl_vec(tsl_s.elements(), 4);
        std::vector<float> cs_min_vec(cs_cache_s.elements(), -1.0);
        std::vector<float> cs_max_vec(cs_cache_s.elements(), 1.0);
        std::vector<float> q_min_vec(query_s.elements(), -1.0);
        std::vector<float> q_max_vec(query_s.elements(), 1.0);
        std::vector<float> kv_min_vec(kv_s.elements(), -1.0);
        std::vector<float> kv_max_vec(kv_s.elements(), 1.0);
        auto k_cache   = mm->add_parameter("k_cache", kv_s);
        auto v_cache   = mm->add_parameter("v_cache", kv_s);
        auto slk       = mm->add_literal(slk_s, slk_vec);
        auto tsl       = mm->add_literal(tsl_s, tsl_vec);
        auto key       = mm->add_literal(0.0f);
        auto value     = mm->add_literal(0.0f);
        auto cs_min    = mm->add_literal(cs_cache_s, cs_min_vec);
        auto cs_max    = mm->add_literal(cs_cache_s, cs_max_vec);
        auto q_min     = mm->add_literal(query_s, q_min_vec);
        auto q_max     = mm->add_literal(query_s, q_max_vec);
        auto kv_min    = mm->add_literal(kv_s, kv_min_vec);
        auto kv_max    = mm->add_literal(kv_s, kv_max_vec);
        auto cos_cache = mm->add_parameter("cos_cache", cs_cache_s);
        auto sin_cache = mm->add_parameter("sin_cache", cs_cache_s);
        query          = mm->add_instruction(migraphx::make_op("clip"), query, q_min, q_max);
        k_cache        = mm->add_instruction(migraphx::make_op("clip"), k_cache, kv_min, kv_max);
        v_cache        = mm->add_instruction(migraphx::make_op("clip"), v_cache, kv_min, kv_max);
        cos_cache      = mm->add_instruction(migraphx::make_op("clip"), cos_cache, cs_min, cs_max);
        sin_cache      = mm->add_instruction(migraphx::make_op("clip"), sin_cache, cs_min, cs_max);

        bool do_rotary           = true;
        std::size_t kv_num_heads = 2;
        int local_window_size    = -1;
        std::size_t num_heads    = 2;
        bool rotary_interleaved  = false;
        float scale              = 1.0;

        const std::size_t batch_size      = query_lens[0];
        const std::size_t sequence_length = query_lens[1];
        std::size_t q_hidden_size         = query_lens[2];
        std::size_t head_size             = q_hidden_size / (num_heads + 2 * kv_num_heads);

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
            rotary_qkv =
                mm->add_instruction(migraphx::make_op("gqa_rotary_embedding",
                                                      {{"kv_num_heads", kv_num_heads},
                                                       {"num_heads", num_heads},
                                                       {"rotary_interleaved", rotary_interleaved}}),
                                    rotary_inputs);
        }

        auto pres_k   = k_cache;
        auto pres_v   = v_cache;
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
        std::vector<migraphx::instruction_ref> concat_k_inputs{rotary_k, slk, pres_k};
        std::vector<migraphx::instruction_ref> concat_v_inputs{rotary_v, slk, pres_v};

        pres_k = mm->add_instruction(
            migraphx::make_op("concat_past_present",
                              {{"kv_num_heads", kv_num_heads}, {"num_heads", num_heads}}),
            concat_k_inputs);
        pres_v = mm->add_instruction(
            migraphx::make_op("concat_past_present",
                              {{"kv_num_heads", kv_num_heads}, {"num_heads", num_heads}}),
            concat_v_inputs);

        // Adding 1 to seq_lens_k, aka past_seq_lens, to allow range literals to start at 0.
        // Putting the add inside the mlir module currently causes an error on their side,
        // so we're leaving it here until that can be solved.
        auto one_lit = mm->add_literal(migraphx::literal{migraphx::shape{slk_s.type(), {1}}, {1}});
        one_lit      = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", slk_s.lens()}}), one_lit);
        auto total_sl = mm->add_instruction(migraphx::make_op("add"), slk, one_lit);

        auto kv_num_heads_factor = num_heads / kv_num_heads;
        auto max_seq_len         = kv_s.lens()[2];
        total_sl                 = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {batch_size, num_heads}}}), total_sl);

        auto q = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {num_heads}}}),
            rotary_qkv);
        auto k = pres_k;
        auto v = pres_v;
        if(kv_num_heads_factor != 1)
        {
            auto kv_new_lens  = kv_lens;
            kv_new_lens.at(1) = num_heads;
            k = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), k);
            v = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), v);
            auto kv_unsqueezed_lens  = kv_lens;
            kv_unsqueezed_lens.at(2) = kv_num_heads_factor;
            k                        = mm->add_instruction(
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
        migraphx::shape range_s{tsl_s.type(), {max_seq_len}};
        auto range = mm->add_literal(range_s, range_vec);
        std::vector<std::size_t> bnsm{batch_size, num_heads, sequence_length, max_seq_len};
        auto bc_range =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", bnsm}}), range);

        auto scalar_s = migraphx::shape{query_s.type(), {1}};
        auto ninf =
            mm->add_literal(migraphx::literal{scalar_s, {-std::numeric_limits<float>::infinity()}});
        ninf = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", bnsm}}), ninf);

        if(migraphx::float_equal(scale, 0.0))
        {
            scale = 1.0f / std::sqrt(static_cast<float>(head_size));
        }
        auto scale_ins = mm->add_literal(migraphx::literal{scalar_s, {scale}});
        scale_ins = mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", bnsm}}),
                                        scale_ins);
        auto mul  = mm->add_instruction(migraphx::make_op("mul"), gemm1, scale_ins);

        if(sequence_length > 1)
        {
            std::vector<int> seq_range_vec(sequence_length);
            std::iota(seq_range_vec.begin(), seq_range_vec.end(), 0);
            migraphx::shape seq_range_s{tsl_s.type(), {sequence_length}};
            auto seq_range = mm->add_literal(seq_range_s, seq_range_vec);
            seq_range      = mm->add_instruction(
                migraphx::make_op("reshape", {{"dims", {sequence_length, 1}}}), seq_range);
            seq_range = mm->add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", bnsm}}), seq_range);
            auto causal_mask =
                mm->add_instruction(migraphx::make_op("greater"), bc_range, seq_range);
            causal_mask = mm->add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}),
                causal_mask);
            mul = mm->add_instruction(migraphx::make_op("where"), causal_mask, ninf, mul);
        }

        auto bc_total_sl = mm->add_instruction(
            migraphx::make_op("reshape", {{"dims", {batch_size, num_heads, 1, 1}}}), total_sl);
        auto mask_comp = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", bnsm}}), bc_total_sl);
        auto mask = mm->add_instruction(migraphx::make_op("greater"), bc_range, mask_comp);
        mask      = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), mask);
        auto where   = mm->add_instruction(migraphx::make_op("where"), mask, ninf, mul);
        auto softmax = mm->add_instruction(migraphx::make_op("softmax", {{"axis", 3}}), where);
        auto scores  = mm->add_instruction(migraphx::make_op("dot"), softmax, v);
        auto out     = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), scores);
        out = mm->add_instruction(
            migraphx::make_op("reshape",
                              {{"dims", {batch_size, sequence_length, head_size * num_heads}}}),
            out);

        mm->add_return({out, pres_k, pres_v});

        return p;
    }
};
