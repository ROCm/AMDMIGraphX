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
#include <migraphx/instruction.hpp>

struct test_sparse_attention_rotary_token_generation_batched
    : verify_program<test_sparse_attention_rotary_token_generation_batched>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        const size_t batch_size                = 2;
        const size_t sequence_length           = 1;
        const size_t num_heads                 = 2;
        const size_t kv_num_heads              = 1;
        const size_t head_size                 = 128;
        const size_t max_cache_sequence_length = 256;
        const size_t sparse_block_size         = 64;
        const size_t num_layouts               = 2;
        const size_t max_blocks                = 4;
        const size_t max_nnz_blocks            = 9;
        const size_t total_sequence_length     = 256;
        const size_t max_rotary_seq_length     = max_cache_sequence_length;
        const size_t rotary_dim                = head_size / 2;
        const float scale                      = 1.0;
        const bool do_rotary                   = true;
        const bool rotary_interleaved          = false;

        migraphx::shape qkv_shape(
            migraphx::shape::float_type,
            {batch_size, num_heads + 2 * kv_num_heads, sequence_length, head_size});
        migraphx::shape past_key_shape(
            migraphx::shape::float_type,
            {batch_size, kv_num_heads, max_cache_sequence_length, head_size});
        migraphx::shape past_value_shape(
            migraphx::shape::float_type,
            {batch_size, kv_num_heads, max_cache_sequence_length, head_size});
        migraphx::shape block_row_indices_shape(migraphx::shape::int32_type,
                                                {num_layouts, max_blocks + 1});
        migraphx::shape block_col_indices_shape(migraphx::shape::int32_type,
                                                {num_layouts, max_nnz_blocks});
        migraphx::shape total_sequence_len_shape(migraphx::shape::int32_type, {1});
        migraphx::shape key_total_sequence_lens_shape(migraphx::shape::int32_type, {batch_size});
        migraphx::shape cos_cache_shape(migraphx::shape::float_type,
                                        {max_rotary_seq_length, rotary_dim});
        migraphx::shape sin_cache_shape(migraphx::shape::float_type,
                                        {max_rotary_seq_length, rotary_dim});

        std::vector<int> bri_val{0, 1, 3, 6, 9, 0, 1, 3, 5, 8};
        std::vector<int> bci_val{0, 0, 1, 0, 1, 2, 0, 2, 3, 0, 0, 1, 1, 2, 1, 2, 3, -1};
        std::vector<int> tsl_val(total_sequence_len_shape.elements(), total_sequence_length);
        std::vector<int> ktsl_val(key_total_sequence_lens_shape.elements(), total_sequence_length);

        auto qkv        = mm->add_parameter("qkv", qkv_shape);
        auto k          = mm->add_literal(0.0f);
        auto v          = mm->add_literal(0.0f);
        auto past_key   = mm->add_parameter("past_key", past_key_shape);
        auto past_value = mm->add_parameter("past_value", past_value_shape);
        auto bri        = mm->add_literal(migraphx::literal{block_row_indices_shape, bri_val});
        auto bci        = mm->add_literal(migraphx::literal{block_col_indices_shape, bci_val});
        auto tsl        = mm->add_literal(migraphx::literal{total_sequence_len_shape, tsl_val});
        auto ktsl = mm->add_literal(migraphx::literal{key_total_sequence_lens_shape, ktsl_val});
        auto cos_cache = mm->add_parameter("cos_cache", cos_cache_shape);
        cos_cache      = add_clip(mm, cos_cache, -1.0f, 1.0f);
        auto sin_cache = mm->add_parameter("sin_cache", sin_cache_shape);
        sin_cache      = add_clip(mm, sin_cache, -1.0f, 1.0f);

        auto sparse_attn =
            mm->add_instruction(migraphx::make_op("sparse_attention",
                                                  {{"do_rotary", do_rotary},
                                                   {"rotary_interleaved", rotary_interleaved},
                                                   {"num_heads", num_heads},
                                                   {"kv_num_heads", kv_num_heads},
                                                   {"scale", scale},
                                                   {"sparse_block_size", sparse_block_size}}),
                                qkv,
                                k,
                                v,
                                past_key,
                                past_value,
                                bri,
                                bci,
                                tsl,
                                ktsl,
                                cos_cache,
                                sin_cache);
        auto attn_output =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sparse_attn);
        auto present_key_output =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), sparse_attn);
        auto present_val_output =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 2}}), sparse_attn);
        mm->add_return({attn_output, present_key_output, present_val_output});

        return p;
    }

    migraphx::instruction_ref
    add_clip(migraphx::module* mod, migraphx::instruction_ref x, float min, float max) const
    {
        auto min_val_lit = mod->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {min}});
        auto max_val_lit = mod->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type, {1}}, {max}});

        auto min_val_lit_bc = mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x->get_shape().lens()}}),
            min_val_lit);
        auto max_val_lit_bc = mod->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x->get_shape().lens()}}),
            max_val_lit);

        return mod->add_instruction(migraphx::make_op("clip"), x, min_val_lit_bc, max_val_lit_bc);
    }
};
