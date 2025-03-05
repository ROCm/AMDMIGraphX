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
#ifndef MIGRAPHX_GUARD_OPERATORS_SPARSE_ATTENTION_HPP
#define MIGRAPHX_GUARD_OPERATORS_SPARSE_ATTENTION_HPP

#include "migraphx/errors.hpp"
#include <algorithm>
#include <limits>
#include <migraphx/op/name.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/shape_for_each.hpp>
#include <migraphx/par_for.hpp>
#include <migraphx/gemm.hpp>
#include <migraphx/argument.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct sparse_attn_parameters
{
    size_t batch_size;
    size_t sequence_length;
    size_t head_size;
    size_t max_cache_sequence_length;
    size_t num_layouts;
    size_t max_blocks;
    size_t max_nnz;
    size_t max_sequence_length;
    size_t total_sequence_length;
    size_t max_rotary_sequence_length;
    size_t rotary_embed_dim;
    bool is_prompt;
    size_t sequence_stride;
    size_t head_stride;
    size_t batch_stride;
    size_t k_offset;
    size_t v_offset;
};

// NOTE ONLY PACKED QKV VARIANT SUPPORTED

// Input shapes:
// 1.  packed qkv         (batch_size, num_heads + 2 * kv_num_heads, sequence_length, head_size)
// 2.  key(opt.)          empty
// 3.  value              empty
// 4.  past_key           (batch_size, kv_num_heads, max_cache_sequence_length, head_size)
// 5.  past_value         (batch_size, kv_num_heads, max_cache_sequence_length, head_size)
// 6.  block_row_indices  (num_layout, max_blocks + 1)
//                         max_blocks = max_sequence_length / sparse_block_size
// 7.  block_col_indices  (num_layout, max_nnz)
// 8.  total_seq_length   (1)
// 9.  key_total_seq_len  (batch_size)
// 10. cos_cache(opt.)    (max_rotaty_seq_length, rotary_dim / 2)
// 11. sin_cache(opt.)    (max_rotaty_seq_length, rotary_dim / 2)
//
// Output shapes:
// 1.  output             (batch_size, sequence_length, num_heads * head_size)
// 2.  present_key        (batch_size, kv_num_heads, max_cache_sequence_length, head_size)
// 3.  present_value      (batch_size, kv_num_heads, max_cache_sequence_length, head_size)
struct sparse_attention : op_name<sparse_attention>
{
    bool do_rotary          = false;
    bool rotary_interleaved = false;
    size_t num_heads;
    size_t kv_num_heads;
    float scale = 0.0f;
    size_t sparse_block_size;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.do_rotary, "do_rotary"),
                    f(self.rotary_interleaved, "rotary_interleaved"),
                    f(self.num_heads, "num_heads"),
                    f(self.kv_num_heads, "kv_num_heads"),
                    f(self.scale, "scale"),
                    f(self.sparse_block_size, "sparse_block_size"));
    }

    shape compute_shape(std::vector<shape> inputs) const
    {
        auto num_inputs          = inputs.size();
        auto expected_num_inputs = do_rotary ? 11 : 9;
        if(num_inputs != expected_num_inputs)
        {
            const std::string do_rotary_s = do_rotary ? "true" : "false";
            MIGRAPHX_THROW("SparseAttention: when do_rotary=" + do_rotary_s +
                           ", number of inputs should be " + std::to_string(expected_num_inputs) +
                           ", is " + std::to_string(num_inputs));
        }

        auto&& qkv_lens = inputs[0].lens();

        return shape(
            {shape{inputs.front().type(), {qkv_lens[0], qkv_lens[2], qkv_lens[3] * num_heads}},
             inputs[3],
             inputs[4]});
    }

    argument compute(const shape& output_shape, const std::vector<argument>& args) const
    {
        auto qkv_arg                        = args[0];
        auto past_key_arg                   = args[3];
        auto past_value_arg                 = args[4];
        auto block_row_indices_arg          = args[5];
        auto block_col_indices_arg          = args[6];
        auto key_total_sequence_lengths_arg = args[8];

        const auto params = make_params(args);

        if(do_rotary)
        {
            auto cos_cache_arg = args[9];
            auto sin_cache_arg = args[10];
            qkv_arg            = rotary_embedding(
                qkv_arg, cos_cache_arg, sin_cache_arg, key_total_sequence_lengths_arg, params);
        }

        // Used to store intermediate results of Q.KT
        argument attention_probs_arg(shape{
            output_shape.sub_shapes()[0].type(),
            {params.batch_size, num_heads, params.sequence_length, params.total_sequence_length}});

        // Used to store attention(Q, K ,V)
        argument output_arg(output_shape.sub_shapes()[0]);

        visit_all(output_arg, attention_probs_arg, qkv_arg, past_key_arg, past_value_arg)(
            [&](auto output, auto attention_probs, auto qkv, auto key_cache, auto value_cache) {
                visit_all(block_row_indices_arg,
                          block_col_indices_arg,
                          key_total_sequence_lengths_arg)([&](auto block_row_indices,
                                                              auto block_col_indices,
                                                              auto key_total_sequence_lengths) {
                    apply_attention(output,
                                    attention_probs,
                                    qkv,
                                    key_cache,
                                    value_cache,
                                    block_row_indices,
                                    block_col_indices,
                                    key_total_sequence_lengths,
                                    params);
                });
            });

        return {{output_arg, past_key_arg, past_value_arg}};
    }

    sparse_attn_parameters make_params(std::vector<argument> args) const
    {
        sparse_attn_parameters params;
        const auto arg_shapes = to_shapes(args);

        params.batch_size            = arg_shapes[0].lens()[0];
        const size_t sequence_length = arg_shapes[0].lens()[2];
        params.sequence_length       = sequence_length;
        const size_t head_size       = arg_shapes[0].lens()[3];
        params.head_size             = head_size;

        params.max_cache_sequence_length = arg_shapes[3].lens()[2];

        params.num_layouts         = arg_shapes[5].lens()[0];
        size_t max_blocks          = arg_shapes[5].lens()[1] - 1;
        params.max_blocks          = max_blocks;
        params.max_sequence_length = max_blocks * sparse_block_size;
        params.max_nnz             = arg_shapes[6].lens()[1];

        params.total_sequence_length = args[7].at<int>(0);

        if(do_rotary)
        {
            params.max_rotary_sequence_length = arg_shapes[9].lens()[0];
            params.rotary_embed_dim           = arg_shapes[9].lens()[1] * 2;
        }

        params.is_prompt       = params.total_sequence_length == sequence_length;
        params.sequence_stride = head_size;
        params.head_stride     = sequence_length * head_size;
        params.batch_stride    = (num_heads + 2 * kv_num_heads) * sequence_length * head_size;
        params.k_offset        = num_heads * sequence_length * head_size;
        params.v_offset        = (num_heads + kv_num_heads) * sequence_length * head_size;

        return params;
    }

    argument rotary_embedding(const argument& qkv_arg,
                              const argument& cos_cache_arg,
                              const argument& sin_cache_arg,
                              const argument& key_total_sequence_lengths_arg,
                              const sparse_attn_parameters& params) const
    {
        std::vector<size_t> pos_ids(params.is_prompt ? 1
                                                     : params.batch_size * params.sequence_length);
        key_total_sequence_lengths_arg.visit([&](auto kts) {
            if(params.is_prompt)
            {
                pos_ids[0] = 0;
            }
            else if(params.sequence_length == 1)
            {
                std::transform(
                    kts.begin(), kts.end(), pos_ids.begin(), [](auto len) { return len - 1; });
            }
            else
            {
                for(int b = 0; b < params.batch_size; ++b)
                {
                    for(int s = 0; s < params.sequence_length; ++s)
                    {
                        pos_ids[b * params.sequence_length + s] =
                            kts[b] - params.sequence_length + s;
                    }
                }
            }
        });

        argument qkv_rotary_arg{qkv_arg.get_shape()};

        visit_all(qkv_rotary_arg, qkv_arg, cos_cache_arg, sin_cache_arg)(
            [&](auto qkv_rotary, auto qkv, auto cos_cache, auto sin_cache) {
                auto q   = qkv.begin();
                auto q_r = qkv_rotary.begin();
                auto k   = q + params.k_offset;
                auto k_r = q_r + params.k_offset;
                auto v   = q + params.v_offset;
                auto v_r = q_r + params.v_offset;

                run_rotary_embedding(q_r,
                                     q,
                                     cos_cache.begin(),
                                     sin_cache.begin(),
                                     rotary_interleaved,
                                     pos_ids.data(),
                                     num_heads,
                                     params);

                run_rotary_embedding(k_r,
                                     k,
                                     cos_cache.begin(),
                                     sin_cache.begin(),
                                     rotary_interleaved,
                                     pos_ids.data(),
                                     kv_num_heads,
                                     params);

                for(auto b = 0u; b < params.batch_size; ++b)
                {
                    auto batch_offset = b * params.batch_stride;
                    std::copy_n(v + batch_offset,
                                kv_num_heads * params.head_size * params.sequence_length,
                                v_r + batch_offset);
                }
            });

        return qkv_rotary_arg;
    }

    template <class T>
    void run_rotary_embedding(T output,
                              T input,
                              T cos_cache,
                              T sin_cache,
                              bool interleaved,
                              const std::size_t* pos_ids,
                              size_t n_heads,
                              const sparse_attn_parameters& params) const
    {
        const std::size_t position_ids_use_batch = params.sequence_length == 1;
        const std::size_t half_rotary_emb_dim    = params.rotary_embed_dim / 2;

        const std::size_t loop_len = params.batch_size * params.sequence_length * n_heads;
        par_for(loop_len, [&](const auto idx) {
            const std::size_t b = (idx / n_heads) / params.sequence_length;
            const std::size_t s = (idx / n_heads) % params.sequence_length;
            const std::size_t n = idx % n_heads;
            const std::size_t block_offset =
                b * params.batch_stride + s * params.sequence_stride + n * params.head_stride;
            auto input_data  = input + block_offset;
            auto output_data = output + block_offset;

            // Cache is (M, H/2) or (M, rotary_embedding_dim/2)
            const std::size_t position_id =
                position_ids_use_batch ? pos_ids[b * params.sequence_length + s] : pos_ids[0] + s;
            const std::size_t cache_offset = position_id * half_rotary_emb_dim;
            auto cos_data                  = cos_cache + cache_offset;
            auto sin_data                  = sin_cache + cache_offset;

            std::size_t cache_idx = 0;
            float sign            = 0.0;
            std::size_t j         = 0;
            for(std::size_t i = 0; i < params.rotary_embed_dim; i++)
            {
                if(interleaved)
                {
                    cache_idx = (i / 2) % half_rotary_emb_dim;
                    sign      = (i % 2 == 0) ? -1.0 : 1.0;
                    j         = (i % 2 == 0) ? i + 1 : i - 1; // i - sign
                }
                else
                {
                    cache_idx = i % half_rotary_emb_dim;
                    sign      = (i < half_rotary_emb_dim) ? -1.0 : 1.0;
                    j         = (i + half_rotary_emb_dim) % params.rotary_embed_dim;
                }
                output_data[i] = input_data[i] * cos_data[cache_idx] +
                                 sign * input_data[j] * sin_data[cache_idx];
            }
            std::copy(input_data + params.rotary_embed_dim,
                      input_data + params.head_size,
                      output_data + params.rotary_embed_dim);
        });
    }

    template <typename T, typename U>
    void apply_attention(T output,
                         T attention_probs,
                         T qkv,
                         T key_cache,
                         T value_cache,
                         U block_row_indices,
                         U block_col_indices,
                         U key_total_sequence_lengths,
                         const sparse_attn_parameters& params) const
    {
        std::vector<bool> is_dense_layout(params.num_layouts);
        for(auto l = 0; l < params.num_layouts; ++l)
            is_dense_layout[l] = this->is_dense_layout(block_row_indices, l);

        const auto kv_head_factor = num_heads / kv_num_heads;

        for(auto batch_idx = 0; batch_idx < params.batch_size; ++batch_idx)
        {
            for(auto head_idx = 0; head_idx < num_heads; ++head_idx)
            {
                const auto kv_head_idx = head_idx / kv_head_factor;

                auto qkv_batch = qkv.begin() + batch_idx * params.batch_stride;
                auto q         = qkv_batch + head_idx * params.head_stride;
                auto k         = qkv_batch + params.k_offset + kv_head_idx * params.head_stride;
                auto v         = qkv_batch + params.v_offset + kv_head_idx * params.head_stride;

                const auto cache_offset = (batch_idx * params.batch_size + head_idx) /
                                          kv_head_factor * params.max_cache_sequence_length *
                                          params.head_size;
                auto k_cache = key_cache.begin() + cache_offset;
                auto v_cache = value_cache.begin() + cache_offset;

                const size_t key_total_sequence_length = key_total_sequence_lengths[batch_idx];
                const size_t past_sequence_length =
                    params.is_prompt ? 0 : key_total_sequence_length - params.sequence_length;

                // Update tail end of k_cache with contents of k. Likewise for v_cache and v.
                // Since one kv_head can correspond to several heads, make sure to only update once.
                if(head_idx % kv_head_factor == 0)
                {
                    std::copy(k,
                              k + params.head_stride,
                              k_cache + past_sequence_length * params.head_size);
                    std::copy(v,
                              v + params.head_stride,
                              v_cache + past_sequence_length * params.head_size);
                }

                const float s   = scale == 0.0f ? 1.0f / std::sqrt(params.head_size) : scale;
                auto attn_probs = attention_probs.begin() +
                                  (batch_idx * params.batch_size + head_idx) *
                                      params.sequence_length * params.max_cache_sequence_length;
                gemm(qkv.get_shape().type(),
                     false,
                     true,
                     params.sequence_length,
                     key_total_sequence_length,
                     params.head_size,
                     s,
                     &q[0],
                     params.head_size,
                     &k_cache[0],
                     params.head_size,
                     0.0f,
                     &attn_probs[0],
                     params.total_sequence_length);

                auto layout_idx = head_idx % params.num_layouts;
                softmax_masked(attn_probs,
                               params.max_cache_sequence_length,
                               params.sequence_length,
                               key_total_sequence_length,
                               past_sequence_length,
                               params.max_blocks,
                               is_dense_layout[layout_idx],
                               layout_idx,
                               block_row_indices,
                               block_col_indices);

                auto out_offset =
                    (batch_idx * params.sequence_length * num_heads + head_idx) * params.head_size;
                auto out = output.begin() + out_offset;
                gemm(qkv.get_shape().type(),
                     false,
                     false,
                     params.sequence_length,
                     params.head_size,
                     key_total_sequence_length,
                     1.0f,
                     &attn_probs[0],
                     params.total_sequence_length,
                     &v_cache[0],
                     params.head_size,
                     0.0f,
                     &out[0],
                     num_heads * params.head_size);
            }
        }
    }

    template <typename T>
    void gemm(shape::type_t dtype,
              bool trans_a,
              bool trans_b,
              size_t m,
              size_t n,
              size_t k,
              float alpha,
              const T* a,
              size_t lda,
              const T* b,
              size_t ldb,
              float beta,
              T* c,
              size_t ldc) const
    {
        std::vector<size_t> a_strides{lda, 1};
        if(trans_a)
            std::swap(a_strides[0], a_strides[1]);
        shape a_shape{dtype, {m, k}, a_strides};

        std::vector<size_t> b_strides{ldb, 1};
        if(trans_b)
            std::swap(b_strides[0], b_strides[1]);
        shape b_shape{dtype, {k, n}, b_strides};

        shape c_shape{dtype, {m, n}, {ldc, 1}};

        auto a_mat = make_view(a_shape, a);
        auto b_mat = make_view(b_shape, b);
        auto c_mat = make_view(c_shape, c);

        migraphx::gemm(c_mat, a_mat, b_mat, alpha, beta);
    }

    template <typename T, typename U>
    // Input qk has shape (sequence_length, key_total_sequence_length), but the underlying
    // buffer has shape (sequence_length, qk_stride), with qk_stride >=
    // key_total_sequence_length
    void softmax_masked(T qk,
                        size_t qk_stride,
                        size_t sequence_length,
                        size_t key_total_sequence_length,
                        size_t past_sequence_length,
                        size_t max_blocks,
                        bool is_dense_layout,
                        size_t layout_idx,
                        U block_row_indices,
                        U block_col_indices) const
    {
        if(is_dense_layout)
        {
            for(size_t q_idx = 0; q_idx < sequence_length; ++q_idx)
            {
                size_t causal_length = past_sequence_length + q_idx + 1;
                row_softmax(qk, causal_length, key_total_sequence_length);
                qk += qk_stride;
            }
            return;
        }

        std::vector<size_t> mask(max_blocks);
        bool apply_mask = false;
        for(size_t q_idx = 0; q_idx < sequence_length; ++q_idx)
        {
            // The Q.KT matrix has shape (sequence_length,
            // key_total_sequence_length), with key_total_sequence_length =
            // past_sequence_length + sequence_length.
            // If past_sequence_length is 0, Q.KT will be a square matrix, if it
            // isn't we pretend otherwise by offseting q_idx
            size_t causal_length = past_sequence_length + q_idx + 1;
            size_t q_abs_idx     = past_sequence_length + q_idx;
            // Expand mask for layout row when q_idx aligns with block edge
            // The mask remains in use for all subsequent q_idx that belong to the same block
            if(q_idx == 0 or q_abs_idx % sparse_block_size == 0)
            {
                size_t layout_row    = q_abs_idx / sparse_block_size;
                size_t col_ind_start = block_row_indices(layout_idx, layout_row);
                size_t col_ind_end   = block_row_indices(layout_idx, layout_row + 1);
                auto nonzero_in_row  = col_ind_end - col_ind_start;
                // Layout is a lower triangular matrix, check if there are any zero
                // elements in current row. If so, expand mask for current row, and
                // mask out the corresponding elements
                apply_mask = nonzero_in_row != layout_row + 1;
                if(apply_mask)
                {
                    auto num_blocks = q_abs_idx / sparse_block_size + 1;
                    std::fill_n(mask.begin(), num_blocks, 0);
                    for(auto i = col_ind_start; i < col_ind_end; ++i)
                        mask[block_col_indices(layout_idx, i)] = 1;
                }
            }

            if(apply_mask)
            {
                auto num_blocks = q_abs_idx / sparse_block_size + 1;
                for(auto i = 0; i < num_blocks; ++i)
                {
                    if(mask[i] == 0)
                        std::fill_n(qk + i * sparse_block_size,
                                    sparse_block_size,
                                    std::numeric_limits<typename T::value_type>::lowest());
                }
            }
            row_softmax(qk, causal_length, key_total_sequence_length);
            qk += qk_stride;
        }
    }

    // Check if there are any zero elements in the lower triangular
    //
    // The block mask matrix is a square (max_blocks, max_blocks) lower triangular matrix
    // (Note: the zeros above the main diagonal correspond to the causal mask)
    // Given that, the maximum number of non-zero elements is max_blocks * (max_blocks + 1) / 2.
    //
    // The last element of block_row_indices for a given layout will always be equal to the
    // actual length of the block_col_indices^, which in turn is equal to the number of non-zero
    // elements in the matrix, that is, in the lower triangular of the matrix.
    //
    // ^The shape of block_col_indices is (num_layouts, max_nnz), with max_nnz being derived
    // from the layout with most non-zero elements. If a layout has fewer than max_nnz non-zero
    // elements, the difference is offset by adding padding. For this reason the second
    // dimension of block_col_indices cannot be used to indicate the number of non-zero elements
    // in a given layout.
    template <typename TV>
    bool is_dense_layout(TV block_row_indices, size_t layout_idx) const
    {
        auto max_blocks              = block_row_indices.get_shape().lens()[1] - 1;
        int32_t num_nonzero_elements = block_row_indices(layout_idx, max_blocks);
        int32_t max_nonzero_elements = max_blocks * (max_blocks + 1) / 2;
        return num_nonzero_elements == max_nonzero_elements;
    }

    template <typename T>
    void row_softmax(T x, size_t causal_length, size_t total_sequence_length) const
    {

        if(causal_length > 0)
        {
            float max = *std::max_element(x, x + causal_length);
            std::transform(x, x + causal_length, x, [max](auto xi) { return expf(xi - max); });
            double sum = std::reduce(x, x + causal_length, 0.0);
            std::transform(x, x + causal_length, x, [sum](auto ei) { return ei / sum; });
        }
        std::fill(x + causal_length, x + total_sequence_length, 0);
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
