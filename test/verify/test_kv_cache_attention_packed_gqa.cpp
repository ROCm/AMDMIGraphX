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

#include "verify_program.hpp"
#include <migraphx/generate.hpp>
#include <migraphx/program.hpp>
#include <migraphx/make_op.hpp>
#include <limits>

// Same starting graph as kv_cache_attention_packed_gqa. The fixed seqlens input
// prevents the random verifier input from masking the entire cache. The reference
// target does not fuse attention. The gpu target does, and splits it when
// MIGRAPHX_FLASH_DECODING_NUM_SPLITS is set (2 divides the cache length of 8).
struct test_kv_cache_attention_packed_gqa
    : verify_program<test_kv_cache_attention_packed_gqa>
{
    migraphx::parameter_map get_inputs() const
    {
        return {{"sl", migraphx::fill_argument({migraphx::shape::int32_type, {1}}, 7)}};
    }

    migraphx::program create_program() const
    {
        // q_heads 4, kv_heads 2, head 4, cache length 8. The packed tensor is Q then K.
        migraphx::program p;
        auto* mm = p.get_main_module();
        migraphx::shape s_half{migraphx::shape::half_type, {1}};
        migraphx::shape s_range{migraphx::shape::int32_type, {8}};
        migraphx::shape s_sl{migraphx::shape::int32_type, {1}};
        migraphx::shape s_rope{migraphx::shape::half_type, {1, 6, 1, 4}};
        migraphx::shape s_cur_v{migraphx::shape::half_type, {1, 2, 1, 4}};
        migraphx::shape s_cache{migraphx::shape::half_type, {1, 2, 8, 4}};

        auto ninf =
            mm->add_literal(migraphx::literal{s_half, {-std::numeric_limits<float>::infinity()}});
        auto scale = mm->add_literal(migraphx::literal{s_half, {0.5}});
        auto range = mm->add_literal(migraphx::literal{s_range, {0, 1, 2, 3, 4, 5, 6, 7}});
        auto sl    = mm->add_parameter("sl", s_sl);
        auto rope  = mm->add_parameter("rope", s_rope);
        auto slc_k = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {4}}, {"ends", {6}}}), rope);
        auto past_k = mm->add_parameter("past_k", s_cache);
        auto cpp_k  = mm->add_instruction(
            migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}), slc_k, sl, past_k);
        auto cur_v  = mm->add_parameter("cur_v", s_cur_v);
        auto past_v = mm->add_parameter("past_v", s_cache);
        auto cpp_v  = mm->add_instruction(
            migraphx::make_op("concat_past_present", {{"kv_num_heads", 2}}), cur_v, sl, past_v);
        auto slc_q = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {4}}}), rope);
        auto unsq_v = mm->add_instruction(
            migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {}}}), cpp_v);
        auto bc_v = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 8, 4}}}), unsq_v);
        auto rsp_v =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 8, 4}}}), bc_v);
        auto unsq_k = mm->add_instruction(
            migraphx::make_op("unsqueeze", {{"axes", {2}}, {"steps", {}}}), cpp_k);
        auto tsp_k = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 1, 2, 4, 3}}}), unsq_k);
        auto bc_k = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 2, 2, 4, 8}}}), tsp_k);
        auto rsp_k =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 4, 8}}}), bc_k);
        auto gemm1 = mm->add_instruction(migraphx::make_op("dot"), slc_q, rsp_k);
        auto bc_ninf =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 4, 1, 8}}}),
                                ninf);
        auto bc_scale =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 4, 1, 8}}}),
                                scale);
        auto cvt_scores = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), gemm1);
        auto cvt_scale = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), bc_scale);
        auto scaled = mm->add_instruction(migraphx::make_op("mul"), cvt_scores, cvt_scale);
        auto bc_sl =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8}}}), sl);
        auto grtr     = mm->add_instruction(migraphx::make_op("greater"), range, bc_sl);
        auto cvt_grtr = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}), grtr);
        auto bc_grtr = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4, 1, 8}}}), cvt_grtr);
        auto cvt_ninf = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), bc_ninf);
        auto mask = mm->add_instruction(migraphx::make_op("where"), bc_grtr, cvt_ninf, scaled);
        auto rdc_max =
            mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {3}}}), mask);
        auto bc_rm = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4, 1, 8}}}), rdc_max);
        auto sub     = mm->add_instruction(migraphx::make_op("sub"), mask, bc_rm);
        auto exp     = mm->add_instruction(migraphx::make_op("exp"), sub);
        auto rdc_sum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3}}}), exp);
        auto bc_rs   = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4, 1, 8}}}), rdc_sum);
        auto div    = mm->add_instruction(migraphx::make_op("div"), exp, bc_rs);
        auto cvt_sm = mm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}), div);
        auto gemm2   = mm->add_instruction(migraphx::make_op("dot"), cvt_sm, rsp_v);
        auto tsp_out = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), gemm2);
        auto rsp_out =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 16}}}), tsp_out);
        mm->add_return({rsp_out, cpp_k, cpp_v});
        return p;
    }
};
