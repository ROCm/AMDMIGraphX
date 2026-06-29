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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/shape.hpp>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

enum class qkv_fomat_t
{
    q_k_v       = 0,
    q_k_v_cross = 1,
    kv_packed   = 2,
    qkv_packed  = 3
};

enum class key_mask_mode_t
{
    none          = -1,
    direct_2d_pad = 0,
    left_pad      = 1,
    right_pad     = 2,
    direct_3d_pad = 3
};

struct multi_head_attention_parameters
{
    int64_t batch_size            = 0;
    int64_t q_sequence_length     = 0;
    int64_t kv_sequence_length    = 0;
    int64_t total_sequence_length = 0;
    int64_t hidden_size           = 0;
    int64_t hidden_size_v         = 0;
    int64_t head_size             = 0;
    int64_t head_size_v           = 0;
    int64_t num_heads             = 0;
    qkv_fomat_t qkv_fomat         = qkv_fomat_t::q_k_v;
    bool qkv_biased               = false;
    float mask_filter_value       = -10000.0f;
    key_mask_mode_t key_pad_mode  = key_mask_mode_t::none;
    /// When true, key/value axis \e n (kv sequence length) is dynamic; reshape uses 0 dims.
    bool kv_sequence_length_dynamic = false;
};

namespace {

shape::dynamic_dimension mha_dim_as_dd(const shape& s, std::size_t i)
{
    if(s.dynamic())
        return s.dyn_dims().at(i);
    const auto v = s.lens().at(i);
    return {v, v};
}

bool mha_dd_equal_or_compatible(const shape::dynamic_dimension& a, const shape::dynamic_dimension& b)
{
    if(a == b)
        return true;
    return a.intersection(b).has_value();
}

void mha_require_fixed_except(const shape& s, std::size_t except_axis)
{
    if(not s.dynamic())
        return;
    for(std::size_t i = 0; i < s.ndim(); ++i)
    {
        if(i == except_axis)
            continue;
        // if(not s.dyn_dims().at(i).is_fixed())
        //     MIGRAPHX_THROW(
        //         "MultiHeadAttention: only key/value sequence length n may be dynamic; axis " +
        //         std::to_string(i) + " must have a fixed size.");
    }
}

int64_t mha_fixed_dim_size(const shape& s, std::size_t i)
{
    const auto dd = mha_dim_as_dd(s, i);
    if(not dd.is_fixed())
        MIGRAPHX_THROW("MultiHeadAttention: expected a fixed-size dimension at axis " +
                       std::to_string(i) + ".");
    return static_cast<int64_t>(dd.get_interval().min);
}

void validate_qkv_separate_dynamic_kv_n(const shape& query_shape,
                                        const shape& key_shape,
                                        const shape& value_shape,
                                        multi_head_attention_parameters& params)
{
    if(query_shape.dynamic() and not std::all_of(query_shape.dyn_dims().begin(), query_shape.dyn_dims().end(), [](const shape::dynamic_dimension& dim) { return dim.is_fixed(); }))
        MIGRAPHX_THROW("MultiHeadAttention: query must be static when key/value sequence length "
                       "n is dynamic.");

    const auto q_seq    = mha_dim_as_dd(query_shape, 1);
    const auto q_hidden = mha_dim_as_dd(query_shape, 2);
    if(not q_seq.is_fixed() or not q_hidden.is_fixed())
        MIGRAPHX_THROW("MultiHeadAttention: query sequence length and hidden size must be static "
                       "when n is dynamic.");

    if(key_shape.dynamic() != value_shape.dynamic())
        MIGRAPHX_THROW("MultiHeadAttention: key and value must both be static or both dynamic for "
                       "the (batch, n, hidden) layout.");

    mha_require_fixed_except(key_shape, 1);
    mha_require_fixed_except(value_shape, 1);

    const auto k_n = mha_dim_as_dd(key_shape, 1);
    const auto v_n = mha_dim_as_dd(value_shape, 1);
    if(not mha_dd_equal_or_compatible(k_n, v_n))
        MIGRAPHX_THROW("MultiHeadAttention: key and value must agree on sequence length n.");

    if(not mha_dd_equal_or_compatible(mha_dim_as_dd(query_shape, 0), mha_dim_as_dd(key_shape, 0)) or
        not mha_dd_equal_or_compatible(mha_dim_as_dd(query_shape, 0), mha_dim_as_dd(value_shape, 0)))
        MIGRAPHX_THROW("MultiHeadAttention: batch_size mismatch between query, key, and value.");

    if(not mha_dd_equal_or_compatible(mha_dim_as_dd(query_shape, 2), mha_dim_as_dd(key_shape, 2)))
        MIGRAPHX_THROW("MultiHeadAttention: hidden size mismatch between query and key.");

    if(not mha_dim_as_dd(value_shape, 2).is_fixed())
        MIGRAPHX_THROW("MultiHeadAttention: value hidden_size_v must be static when n is dynamic.");

    params.batch_size        = mha_fixed_dim_size(query_shape, 0);
    params.q_sequence_length = mha_fixed_dim_size(query_shape, 1);
    params.hidden_size       = mha_fixed_dim_size(query_shape, 2);
    params.head_size         = params.hidden_size / params.num_heads;
    params.hidden_size_v     = mha_fixed_dim_size(value_shape, 2);
    params.head_size_v      = params.hidden_size_v / params.num_heads;
    params.qkv_fomat         = qkv_fomat_t::q_k_v;

    if(not k_n.is_fixed())
    {
        params.kv_sequence_length_dynamic = true;
        params.kv_sequence_length         = 0;
    }
    else
    {
        params.kv_sequence_length_dynamic = false;
        params.kv_sequence_length         = mha_fixed_dim_size(key_shape, 1);
    }
}

} // namespace

struct parse_multi_head_attention : op_parser<parse_multi_head_attention>
{

    std::vector<op_desc> operators() const { return {{"MultiHeadAttention"}}; }

    void unpack_qkv(const onnx_parser::node_info& info,
                    instruction_ref& query,
                    instruction_ref& key,
                    instruction_ref& value) const
    {
        // (batch_size, q_sequence_length, num_heads, 3, head_size) ->
        // (3, batch_size, q_sequence_length, num_heads, head_size)
        auto qkv_packed =
            info.add_instruction(make_op("transpose", {{"permutation", {3, 0, 1, 2, 4}}}), query);
        query = info.add_instruction(
            make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), qkv_packed);
        query = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), query);
        key   = info.add_instruction(
            make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), qkv_packed);
        key   = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), key);
        value = info.add_instruction(
            make_op("slice", {{"axes", {0}}, {"starts", {2}}, {"ends", {3}}}), qkv_packed);
        value = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), value);
    }

    void unpack_kv(const onnx_parser::node_info& info,
                   instruction_ref& key,
                   instruction_ref& value) const
    {
        // (batch_size, kv_sequence_length, num_heads, 2, head_size) ->
        // (2, batch_size, kv_sequence_length, num_heads, head_size)
        auto kv_packed =
            info.add_instruction(make_op("transpose", {{"permutation", {3, 0, 1, 2, 4}}}), key);
        key = info.add_instruction(
            make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {1}}}), kv_packed);
        key   = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), key);
        value = info.add_instruction(
            make_op("slice", {{"axes", {0}}, {"starts", {1}}, {"ends", {2}}}), kv_packed);
        value = info.add_instruction(make_op("squeeze", {{"axes", {0}}}), value);
    }

    void check_query_dim(const std::vector<instruction_ref>& args,
                         multi_head_attention_parameters& params) const
    {
        const auto& query_shape = args[0]->get_shape();
        const auto query_dim    = query_shape.ndim();

        if(query_shape.dynamic() and not std::all_of(query_shape.dyn_dims().begin(), query_shape.dyn_dims().end(), [](const shape::dynamic_dimension& dim) { return dim.is_fixed(); }))
            MIGRAPHX_THROW("MultiHeadAttention: dynamic query is not supported.");

        // const auto query_lens = query_shape.lens();
        auto query_dyn_shape = query_shape.to_dynamic();
        auto query_dyn_dims = query_dyn_shape.dyn_dims();
        std::vector<std::size_t> query_lens(query_dyn_dims.size());
        if(std::all_of(query_dyn_dims.begin(), query_dyn_dims.end(), [](const shape::dynamic_dimension& dim) { return dim.is_fixed(); }))
        {
            std::transform(query_dyn_dims.begin(), query_dyn_dims.end(), query_lens.begin(), [](const shape::dynamic_dimension& dim) { return dim.get_interval().max; });
        }
        else
        {
            MIGRAPHX_THROW("MultiHeadAttention: Input must be fixed if dynamic");
        }

        params.kv_sequence_length_dynamic = false;
        params.batch_size                 = query_lens[0];
        params.q_sequence_length          = query_lens[1];

        if(query_dim != 3 and query_dim != 5)
            MIGRAPHX_THROW("MultiHeadAttention: Input 'query' rank needs to be 3 or 5, current: " +
                           std::to_string(query_dim));

        if(query_dim == 5)
        {
            if(query_lens[2] != params.num_heads or query_lens[3] != 3)
                MIGRAPHX_THROW("MultiHeadAttention: Input 'query' shape needs to be (batch_size, "
                               "q_sequence_length, num_heads, 3, head_size) for packed input.");

            params.kv_sequence_length = query_lens[1];
            params.head_size          = query_lens[4];
            params.head_size_v        = query_lens[4];
            params.hidden_size        = params.num_heads * query_lens[4];
            params.hidden_size_v      = params.num_heads * query_lens[4];
            params.qkv_fomat          = qkv_fomat_t::qkv_packed;
        }
        else // query_dim == 3
        {
            if(args.size() < 2)
                MIGRAPHX_THROW("MultiHeadAttention: Wrong number of inputs, 'key' is missing.");

            params.hidden_size = query_lens[2];
            params.head_size   = query_lens[2] / params.num_heads;

            const auto& key_shape = args[1]->get_shape();
            const auto key_dim    = key_shape.ndim();

            if(key_shape.dynamic() and key_dim != 3)
                MIGRAPHX_THROW("MultiHeadAttention: dynamic key is only supported for rank-3 "
                               "(batch, n, hidden_size) inputs.");

            if(key_dim < 3 or key_dim > 5)
                MIGRAPHX_THROW(
                    "MultiHeadAttention: Input 'key' rank needs to be 3, 4 or 5, current: " +
                    std::to_string(key_dim));

            if(key_dim == 5)
            {
                if(key_shape.dynamic())
                    MIGRAPHX_THROW("MultiHeadAttention: packed key (rank 5) does not support "
                                   "dynamic shapes.");

                const auto key_lens = key_shape.lens();
                if(key_lens[0] != params.batch_size or key_lens[2] != params.num_heads or
                   key_lens[3] != 2 or key_lens[4] != params.head_size)
                    MIGRAPHX_THROW("MultiHeadAttention: Input 'key' shape needs to be (batch_size, "
                                   "kv_sequence_length, num_heads, 2, head_size)");

                params.kv_sequence_length = key_lens[1];
                params.hidden_size_v      = params.hidden_size;
                params.head_size_v        = key_lens[4];
                params.qkv_fomat          = qkv_fomat_t::kv_packed;
            }
            else
            {
                if(args.size() < 3)
                    MIGRAPHX_THROW(
                        "MultiHeadAttention: Wrong number of inputs, 'value' is missing.");

                const auto& value_shape = args[2]->get_shape();
                const auto value_dim    = value_shape.ndim();

                if(value_shape.dynamic() and value_dim != 3)
                    MIGRAPHX_THROW("MultiHeadAttention: dynamic value is only supported for rank-3 "
                                   "(batch, n, hidden_size_v) inputs.");

                if(key_dim != value_dim)
                    MIGRAPHX_THROW(
                        "MultiHeadAttention: Input 'key' and 'value' rank needs to be equal.");

                if(key_dim == 3)
                {
                    const bool kv_any_dynamic = key_shape.dynamic() or value_shape.dynamic();
                    if(kv_any_dynamic)
                    {
                        validate_qkv_separate_dynamic_kv_n(
                            query_shape, key_shape, value_shape, params);
                    }
                    else
                    {
                        const auto key_lens   = key_shape.lens();
                        const auto value_lens = value_shape.lens();
                        if(key_lens[0] != params.batch_size or key_lens[2] != params.hidden_size)
                            MIGRAPHX_THROW("MultiHeadAttention: Input 'key' shape needs to be "
                                           "(batch_size, kv_sequence_length, hidden_size)");

                        if(value_lens[0] != params.batch_size or value_lens[1] != key_lens[1])
                            MIGRAPHX_THROW("MultiHeadAttention: Input 'value' shape needs to be "
                                           "(batch_size, kv_sequence_length, hidden_size_v)");

                        params.kv_sequence_length = key_lens[1];
                        params.hidden_size_v      = value_lens[2];
                        params.head_size_v        = value_lens[2] / params.num_heads;
                        params.qkv_fomat          = qkv_fomat_t::q_k_v;
                    }
                }
                else // key_dim == 4
                {
                    if(key_shape.dynamic() or value_shape.dynamic())
                        MIGRAPHX_THROW("MultiHeadAttention: dynamic shapes are not supported for "
                                       "rank-4 (cross) key/value layout.");

                    const auto key_lens   = key_shape.lens();
                    const auto value_lens = value_shape.lens();
                    if(key_lens[0] != params.batch_size or key_lens[1] != params.num_heads or
                       key_lens[3] != params.head_size)
                        MIGRAPHX_THROW("MultiHeadAttention: Input 'key' shape needs to be "
                                       "(batch_size, num_heads, kv_sequence_length, head_size)");

                    if(value_lens[0] != params.batch_size or value_lens[1] != params.num_heads or
                       value_lens[2] != key_lens[2])
                        MIGRAPHX_THROW("MultiHeadAttention: Input 'value' shape needs to be "
                                       "(batch_size, num_heads, kv_sequence_length, head_size_v)");

                    params.kv_sequence_length = key_lens[2];
                    params.hidden_size_v      = value_lens[3] * params.num_heads;
                    params.head_size_v        = value_lens[3];
                    params.qkv_fomat          = qkv_fomat_t::q_k_v_cross;
                }
            }
        }
    }

    void check_key_padding_mask(const std::vector<instruction_ref>& args,
                                multi_head_attention_parameters& params) const
    {
        if(args.size() > 4)
        {
            if(params.kv_sequence_length_dynamic)
            {
                const auto& ms = args.at(4)->get_shape();
                if(ms.dynamic())
                    MIGRAPHX_THROW("MultiHeadAttention: dynamic key_padding_mask is not supported "
                                   "when key/value sequence length n is dynamic.");
                if(ms.elements() != 0)
                    MIGRAPHX_THROW("MultiHeadAttention: key_padding_mask is not supported when "
                                   "key/value sequence length n is dynamic.");
                return;
            }

            // Skip validation if the mask is empty (optional input not provided)
            if(args.at(4)->get_shape().elements() == 0)
                return;

            const auto key_pad_lens     = args.at(4)->get_shape().lens();
            const auto key_pad_len_size = key_pad_lens.size();
            const auto key_pad_type     = args.at(4)->get_shape().type();

            if(key_pad_type != shape::int32_type)
                MIGRAPHX_THROW("MultiHeadAttention: Key padding mask must be a int32 tensor");

            if(key_pad_len_size > 3 or key_pad_len_size < 1)
                MIGRAPHX_THROW(
                    "MultiHeadAttention: Key_pad_mask must be either 1D, 2D or 3D shape tensor");

            if(key_pad_len_size == 1)
            {
                auto key_pad_batch = key_pad_lens.at(0);
                // For left padding mode, the key padding mask is expected to have a size of (3 *
                // batch_size + 2). This format is required by certain ONNX models for compatibility
                // with specific left-padding implementations.
                if(key_pad_batch != params.batch_size and
                   key_pad_batch != (3 * params.batch_size + 2))
                    MIGRAPHX_THROW("MultiHeadAttention: Key Padding Mask must be either batch or 3 "
                                   "x Batch + 2 for 1D key pads");

                if(key_pad_batch == params.batch_size)
                {
                    params.key_pad_mode = key_mask_mode_t::right_pad;
                }
                else
                {
                    params.key_pad_mode = key_mask_mode_t::left_pad;
                }
            }
            else if(key_pad_len_size == 2)
            {
                const auto key_pad_batch         = key_pad_lens.at(0);
                const auto key_pad_total_seq_len = key_pad_lens.at(1);

                if(key_pad_batch != params.batch_size or
                   key_pad_total_seq_len != params.kv_sequence_length)
                {
                    MIGRAPHX_THROW("MultiHeadAttention: 2D Keypad mask must have either (batch, "
                                   "kv_sequence_length) or (batch, total_sequence_length)");
                }
                params.key_pad_mode = key_mask_mode_t::direct_2d_pad;
            }
            else // key_pad_len_size == 3 here
            {
                const auto key_pad_batch         = key_pad_lens.at(0);
                const auto key_pad_seq_len       = key_pad_lens.at(1);
                const auto key_pad_total_seq_len = key_pad_lens.at(2);
                if(key_pad_batch != params.batch_size or
                   key_pad_seq_len != params.kv_sequence_length or
                   key_pad_total_seq_len != params.total_sequence_length)
                {
                    MIGRAPHX_THROW("MultiHeadAttention: 3D Keypad mask must have either (batch, "
                                   "kv_sequence_length, total_sequence_length) or (batch, "
                                   "total_sequence_length)");
                }
                params.key_pad_mode = key_mask_mode_t::direct_3d_pad;
            }
        }
    }

    void check_bias(const std::vector<instruction_ref>& args,
                    multi_head_attention_parameters& params) const
    {
        if(args.size() > 3)
        {
            auto bias      = args.at(3);
            auto bias_lens = bias->get_shape().lens();

            if(bias_lens.size() == 1)
            {
                if(params.kv_sequence_length_dynamic)
                    MIGRAPHX_THROW("MultiHeadAttention: QKV bias is not supported when key/value "
                                   "sequence length n is dynamic.");
                if(bias_lens[0] != params.hidden_size_v + (2 * params.hidden_size))
                    MIGRAPHX_THROW(
                        "MultiheadAttention: 1D Bias must be of size hidden_size + hidden_size "
                        "+ v_hidden_size");
                params.qkv_biased = true;
            }
            else
            {
                // For other bias shapes, skip bias processing but don't throw error
                params.qkv_biased = false;
            }
        }
    }

    void check_attention_bias(const std::vector<instruction_ref>& args,
                              const multi_head_attention_parameters& params) const
    {
        if(args.size() > 5)
        {
            if(params.kv_sequence_length_dynamic)
            {
                const auto& bs = args.at(5)->get_shape();
                // if(bs.dynamic())
                //     MIGRAPHX_THROW("MultiHeadAttention: dynamic attention_bias is not supported when "
                //                    "key/value sequence length n is dynamic.");
                // if(bs.elements() != 0)
                //     MIGRAPHX_THROW("MultiHeadAttention: attention_bias is not supported when "
                //                    "key/value sequence length n is dynamic.");
                return;
            }

            const auto attn_bias_lens = args.at(5)->get_shape().lens();

            if(attn_bias_lens.size() != 4)
                MIGRAPHX_THROW("MultiHeadAttention: attention_bias must be 4D shape");

            // attention_bias shape: (batch_size, num_heads, sequence_length, total_sequence_length)
            if(attn_bias_lens[0] != params.batch_size)
                MIGRAPHX_THROW(
                    "MultiHeadAttention: attention_bias first dimension must be batch_size");

            if(attn_bias_lens[1] != params.num_heads)
                MIGRAPHX_THROW(
                    "MultiHeadAttention: attention_bias second dimension must be num_heads");

            if(attn_bias_lens[2] != params.q_sequence_length)
                MIGRAPHX_THROW(
                    "MultiHeadAttention: attention_bias third dimension must be sequence_length");

            if(attn_bias_lens[3] != params.kv_sequence_length)
                MIGRAPHX_THROW("MultiHeadAttention: attention_bias fourth dimension must be "
                               "total_sequence_length");
        }
    }

    void check_past_key(const std::vector<instruction_ref>& args,
                        const multi_head_attention_parameters& params) const
    {
        if(args.size() <= 6)
            return;

        if(params.kv_sequence_length_dynamic)
        {
            const auto& ps = args.at(6)->get_shape();
            if(not ps.dynamic() and ps.elements() > 0)
                MIGRAPHX_THROW("MultiHeadAttention: past_key is not supported when key/value "
                               "sequence length n is dynamic.");
            return;
        }

        // Skip validation if past_key is empty (optional input not provided)
        if(args.at(6)->get_shape().elements() == 0)
            return;

        const auto past_key_lens = args.at(6)->get_shape().lens();
        if(past_key_lens.size() != 4)
            MIGRAPHX_THROW("MultiHeadAttention: past_key must be 4D shape");

        if(past_key_lens[0] != params.batch_size)
            MIGRAPHX_THROW("MultiHeadAttention: past_key first dimension must be batch_size");

        if(past_key_lens[1] != params.num_heads)
            MIGRAPHX_THROW("MultiHeadAttention: past_key second dimension must be num_heads");

        if(past_key_lens[3] != params.head_size)
            MIGRAPHX_THROW("MultiHeadAttention: past_key fourth dimension must be head_size");
    }

    void check_past_value(const std::vector<instruction_ref>& args,
                          const multi_head_attention_parameters& params) const
    {
        if(args.size() <= 7)
            return;

        if(params.kv_sequence_length_dynamic)
        {
            const auto& ps = args.at(7)->get_shape();
            if(not ps.dynamic() and ps.elements() > 0)
                MIGRAPHX_THROW("MultiHeadAttention: past_value is not supported when key/value "
                               "sequence length n is dynamic.");
            return;
        }

        // Skip validation if past_value is empty (optional input not provided)
        if(args.at(7)->get_shape().elements() == 0)
            return;

        const auto past_value_lens = args.at(7)->get_shape().lens();
        if(past_value_lens.size() != 4)
            MIGRAPHX_THROW("MultiHeadAttention: past_value must be 4D shape");

        if(past_value_lens[0] != params.batch_size)
            MIGRAPHX_THROW("MultiHeadAttention: past_value first dimension must be batch_size");

        if(past_value_lens[1] != params.num_heads)
            MIGRAPHX_THROW("MultiHeadAttention: past_value second dimension must be num_heads");

        if(past_value_lens[3] != params.head_size_v)
            MIGRAPHX_THROW("MultiHeadAttention: past_value fourth dimension must be head_size_v");
    }

    void check_past_key_value_match(const std::vector<instruction_ref>& args) const
    {
        if(args.size() <= 7)
            return;

        // Skip if either past_key or past_value is empty
        if(args.at(6)->get_shape().elements() == 0 or args.at(7)->get_shape().elements() == 0)
            return;

        const auto past_key_lens   = args.at(6)->get_shape().lens();
        const auto past_value_lens = args.at(7)->get_shape().lens();
        if(past_value_lens[2] != past_key_lens[2])
            MIGRAPHX_THROW("MultiHeadAttention: past_key and past_value must have "
                           "matching past_sequence_length");
    }

    void check_past_sequence_length(const std::vector<instruction_ref>& args) const
    {
        if(args.size() <= 8)
            return;

        // Skip validation if past_sequence_length is empty
        if(args.at(8)->get_shape().elements() == 0)
            return;

        const auto past_seq_len_type = args.at(8)->get_shape().type();
        if(past_seq_len_type != shape::int32_type)
            MIGRAPHX_THROW("MultiHeadAttention: past_sequence_length must be a int32 tensor");
    }

    void check_past_inputs(const std::vector<instruction_ref>& args,
                           const multi_head_attention_parameters& params) const
    {
        check_past_key(args, params);
        check_past_value(args, params);
        check_past_key_value_match(args);
        check_past_sequence_length(args);
    }

    void check_inputs(const std::vector<instruction_ref>& args,
                      multi_head_attention_parameters& params) const
    {
        if(args.empty() or args.size() > 9)
            MIGRAPHX_THROW(
                "MultiHeadAttention: Wrong number of inputs. Only 'query', 'key', "
                "'value', bias, key_padding_mask, attention_bias, past_key, past_value, and "
                "past_sequence_length inputs are supported.");

        // Order matters here. Most parameters defined by input query, key, value parameters
        // This must be used first to extract hidden size, batch, etc
        check_query_dim(args, params);
        check_bias(args, params);
        check_key_padding_mask(args, params);
        check_attention_bias(args, params);
        check_past_inputs(args, params);
    }

    std::tuple<instruction_ref, instruction_ref, instruction_ref>
    apply_qkv_bias(const onnx_parser::node_info& info,
                   const multi_head_attention_parameters& params,
                   instruction_ref bias,
                   instruction_ref query,
                   instruction_ref key,
                   instruction_ref value) const
    {
        if(params.kv_sequence_length_dynamic)
            MIGRAPHX_THROW("MultiHeadAttention: QKV bias is not supported when key/value sequence "
                           "length n is dynamic.");

        auto bias_unsq = info.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), bias);
        bias_unsq      = info.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), bias_unsq);
        // slice out piece of bias data for each qkv

        auto q_bias = info.add_instruction(
            make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {params.hidden_size}}}),
            bias_unsq);
        auto q_bias_bc = info.add_instruction(
            make_op(
                "multibroadcast",
                {{"out_lens", {params.batch_size, params.q_sequence_length, params.hidden_size}}}),
            q_bias);

        auto k_bias    = info.add_instruction(make_op("slice",
                                                      {{"axes", {2}},
                                                       {"starts", {params.hidden_size}},
                                                       {"ends", {2 * params.hidden_size}}}),
                                           bias_unsq);
        auto k_bias_bc = info.add_instruction(
            make_op(
                "multibroadcast",
                {{"out_lens", {params.batch_size, params.kv_sequence_length, params.hidden_size}}}),
            k_bias);

        auto v_bias = info.add_instruction(
            make_op("slice",
                    {{"axes", {2}},
                     {"starts", {2 * params.hidden_size}},
                     {"ends", {2 * params.hidden_size + params.hidden_size_v}}}),
            bias_unsq);
        auto v_bias_bc = info.add_instruction(
            make_op("multibroadcast",
                    {{"out_lens",
                      {params.batch_size, params.kv_sequence_length, params.hidden_size_v}}}),
            v_bias);

        // Need to reshape this to get back to hidden dimension since biases are shaped with respect
        // to each qkv hidden dimension length
        auto q_reshaped = info.add_instruction(make_op("reshape",
                                                       {{"dims",
                                                         {params.batch_size,
                                                          params.q_sequence_length,
                                                          params.num_heads * params.head_size}}}),
                                               query);
        auto k_reshaped = info.add_instruction(make_op("reshape",
                                                       {{"dims",
                                                         {params.batch_size,
                                                          params.kv_sequence_length,
                                                          params.num_heads * params.head_size}}}),
                                               key);
        auto v_reshaped = info.add_instruction(make_op("reshape",
                                                       {{"dims",
                                                         {params.batch_size,
                                                          params.kv_sequence_length,
                                                          params.num_heads * params.head_size_v}}}),
                                               value);

        auto biased_query = info.add_instruction(make_op("add"), q_bias_bc, q_reshaped);
        auto biased_key   = info.add_instruction(make_op("add"), k_bias_bc, k_reshaped);
        auto biased_value = info.add_instruction(make_op("add"), v_bias_bc, v_reshaped);

        // reshape this back out to (Batch, sequence_length, num_heads, head_size) once we've done
        // the proper add.
        biased_query = info.add_instruction(make_op("reshape",
                                                    {{"dims",
                                                      {params.batch_size,
                                                       params.q_sequence_length,
                                                       params.num_heads,
                                                       params.head_size}}}),
                                            biased_query);
        biased_key   = info.add_instruction(make_op("reshape",
                                                    {{"dims",
                                                      {params.batch_size,
                                                       params.kv_sequence_length,
                                                       params.num_heads,
                                                       params.head_size}}}),
                                          biased_key);
        biased_value = info.add_instruction(make_op("reshape",
                                                    {{"dims",
                                                      {params.batch_size,
                                                       params.kv_sequence_length,
                                                       params.num_heads,
                                                       params.head_size_v}}}),
                                            biased_value);

        return std::make_tuple(biased_query, biased_key, biased_value);
    }

    std::tuple<instruction_ref, instruction_ref, instruction_ref>
    handle_qkv_packing(const onnx_parser::node_info& info,
                       const multi_head_attention_parameters& params,
                       const std::vector<instruction_ref>& args) const
    {
        auto query = args[0];
        instruction_ref key;
        instruction_ref value;

        if(params.qkv_fomat == qkv_fomat_t::qkv_packed)
        {
            // Packed QKV: (batch_size, q_sequence_length, num_heads, 3, head_size)
            unpack_qkv(info, query, key, value);
        }
        else
        {
            // Query: (batch_size, q_sequence_length, hidden_size)
            // Use 0 dims so batch and q_sequence_length follow the input (needed when key/value n
            // is dynamic and shapes are inferred from separate inputs).
            std::vector<int64_t> q_dims{0, 0, params.num_heads, params.head_size};
            query = info.add_instruction(make_op("reshape", {{"dims", q_dims}}), query);

            key = args[1];

            if(params.qkv_fomat == qkv_fomat_t::kv_packed)
            {
                // Packed KV: (batch_size, kv_sequence_length, num_heads, 2, head_size)
                unpack_kv(info, key, value);
            }
            else
            {
                value = args[2];
                if(params.qkv_fomat == qkv_fomat_t::q_k_v)
                {
                    // Key: (batch_size, kv_sequence_length, hidden_size)
                    // Value: (batch_size, kv_sequence_length, hidden_size_v)
                    // 0 copies batch and kv sequence length n from inputs (supports dynamic n).
                    std::vector<int64_t> k_dims{0, 0, params.num_heads, params.head_size};
                    std::vector<int64_t> v_dims{0, 0, params.num_heads, params.head_size_v};
                    key   = info.add_instruction(make_op("reshape", {{"dims", k_dims}}), key);
                    value = info.add_instruction(make_op("reshape", {{"dims", v_dims}}), value);
                }
            }
        }

        return std::make_tuple(query, key, value);
    }

    // Slice, mul, convert and concat until we get a mask matrix useful prior to the where
    instruction_ref
    generate_raw_mask_per_batch(const onnx_parser::node_info& info,
                                const instruction_ref mask_index,
                                const migraphx::shape input_shape,
                                const multi_head_attention_parameters& attention) const
    {
        auto batch_size    = attention.batch_size;
        auto total_seq_len = attention.kv_sequence_length;
        auto num_heads     = attention.num_heads;

        // Other two cases require us to generate masks from sequence or total sequence length pads.
        auto pass_value_lit =
            info.add_literal(migraphx::literal{migraphx::shape{input_shape.type(), {1}, {1}}, {0}});
        auto mask_value_lit = info.add_literal(migraphx::literal{
            migraphx::shape{input_shape.type(), {1}, {1}}, {attention.mask_filter_value}});

        // For dim = 2 or dim =3 generate the appropriate mask across batches
        // We need to handle the batch case since raw masking involves shape [batch, seq_len] or
        // [batch, seq_len, total_seq_len],
        auto bc_pass = info.add_instruction(
            make_op("multibroadcast",
                    {{"out_lens", {batch_size, num_heads, total_seq_len, total_seq_len}}}),
            pass_value_lit);
        auto bc_mask = info.add_instruction(
            make_op("multibroadcast",
                    {{"out_lens", {batch_size, num_heads, total_seq_len, total_seq_len}}}),
            mask_value_lit);

        // For raw masks we just need to mask out key value padding thus the 3d mask isn't needed
        // here.
        auto raw_mask = info.add_instruction(
            make_op("reshape", {{"dims", {batch_size, 1, 1, total_seq_len}}}), mask_index);
        raw_mask = info.add_instruction(
            make_op("multibroadcast",
                    {{"out_lens", {batch_size, num_heads, total_seq_len, total_seq_len}}}),
            raw_mask);
        raw_mask = info.add_instruction(
            make_op("reshape", {{"dims", {batch_size, num_heads, total_seq_len, total_seq_len}}}),
            raw_mask);

        // Reuse "0" broadcasted converted to int32 to check if input mask is greater than 0 for
        // where condition
        auto in_pass =
            info.add_instruction(make_op("convert", {{"target_type", shape::int32_type}}), bc_pass);
        auto in_bool = info.add_instruction(make_op("equal"), raw_mask, in_pass);
        in_bool      = info.add_instruction(
            make_op("convert", {{"target_type", migraphx::shape::bool_type}}), in_bool);
        return info.add_instruction(make_op("where"), in_bool, bc_mask, bc_pass);
    }

    // Convert per batch right padding values and generate raw mask
    // Used so we can leverage
    instruction_ref
    get_raw_mask_from_right_padding(const onnx_parser::node_info& info,
                                    const instruction_ref right_mask,
                                    const multi_head_attention_parameters& attention) const
    {
        auto batch_size    = attention.batch_size;
        auto kv_seq_length = attention.kv_sequence_length;

        // Gen list of indices to compare to the exclusive start of right padding
        std::vector<size_t> indices_vec(kv_seq_length, 0);
        std::iota(indices_vec.begin(), indices_vec.end(), 0);
        auto indices    = info.add_literal(migraphx::literal{
            migraphx::shape{migraphx::shape::int32_type, {static_cast<size_t>(kv_seq_length)}, {1}},
            indices_vec});
        auto indices_bc = info.add_instruction(
            make_op("multibroadcast", {{"out_lens", {batch_size, kv_seq_length}}}), indices);
        auto right_mask_bc = info.add_instruction(
            make_op("multibroadcast", {{"out_lens", {batch_size, kv_seq_length}}}), right_mask);
        auto in_bool = info.add_instruction(make_op("less"), indices_bc, right_mask_bc);

        return info.add_instruction(
            make_op("convert", {{"target_type", migraphx::shape::int32_type}}), in_bool);
    }

    std::optional<instruction_ref>
    create_input_mask(const onnx_parser::node_info& info,
                      const instruction_ref mask_index,
                      const migraphx::shape input_shape,
                      const multi_head_attention_parameters& attention) const
    {
        // Shape Scale dot attention prior to mask will be in (batch, num_heads, query_size,
        // query_size) thus mask needs to handle batch and query_size We should return mask of
        // batch, 1, query_size, query_size so that this per-batch masked can be broadcasted across
        // each attention head

        if((attention.key_pad_mode == key_mask_mode_t::direct_2d_pad) or
           (attention.key_pad_mode == key_mask_mode_t::direct_3d_pad))
        { // Raw Mask - 0 means mask, 1 means pass through. Apply mask_filter_val to mask indices
          // and zero otherwise
            // Need to generate from 2 dims or 3 dim cases
            return generate_raw_mask_per_batch(info, mask_index, input_shape, attention);
        }
        else if(attention.key_pad_mode == key_mask_mode_t::right_pad)
        {
            auto right_mask = get_raw_mask_from_right_padding(info, mask_index, attention);
            return generate_raw_mask_per_batch(info, right_mask, input_shape, attention);
        }

        return nullopt;
    }

    multi_head_attention_parameters handle_attributes(const onnx_parser::node_info& info,
                                                      const onnx_parser& parser) const
    {
        if(not contains(info.attributes, "num_heads"))
            MIGRAPHX_THROW("MultiHeadAttention: num_heads attribute is required");

        multi_head_attention_parameters params;
        params.num_heads = parser.parse_value(info.attributes.at("num_heads")).at<int>();

        if(contains(info.attributes, "mask_filter_value"))
        {
            params.mask_filter_value =
                parser.parse_value(info.attributes.at("mask_filter_value")).at<float>();
        }
        else
        {
            params.mask_filter_value = -10000.0f;
        }

        return params;
    }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        auto params = handle_attributes(info, parser);
        check_inputs(args, params);

        // Handle packing mode of qkv inputs
        // Output should be (batch, sequence_length, num_heads, head_size) for each QKV
        // Depending on attention mode these will change the sequence length / hidden size
        auto [query, key, value] = handle_qkv_packing(info, params, args);

        // Apply bias to QKV inputs after unpacking
        if(params.qkv_biased)
        {
            auto bias = args[3];
            auto [biased_query, biased_key, biased_value] =
                apply_qkv_bias(info, params, bias, query, key, value);

            query = biased_query;
            key   = biased_key;
            value = biased_value;
        }

        // Target shape: (batch_size, num_heads, sequence_length, head_size)
        std::vector<int64_t> perm{0, 2, 1, 3};
        query = info.add_instruction(make_op("transpose", {{"permutation", perm}}), query);
        if(params.qkv_fomat != qkv_fomat_t::q_k_v_cross)
        {
            key   = info.add_instruction(make_op("transpose", {{"permutation", perm}}), key);
            value = info.add_instruction(make_op("transpose", {{"permutation", perm}}), value);
        }

        // Handle past_key and past_value concatenation using concat_past_present
        std::optional<instruction_ref> present_key;
        std::optional<instruction_ref> present_value;
        if(args.size() > 7)
        {
            auto past_key   = args[6];
            auto past_value = args[7];

            // Only use concat_past_present if past states are non-empty
            if(past_key->get_shape().elements() > 0 and past_value->get_shape().elements() > 0)
            {
                // If past_sequence_length is provided (input 8), use it, otherwise use batch-wise
                // zeros
                instruction_ref seqlens_k;
                if(args.size() > 8 and args[8]->get_shape().elements() > 0)
                {
                    seqlens_k = args[8];
                }
                else
                {
                    std::vector<int32_t> zeros(params.batch_size, 0);
                    seqlens_k = info.add_literal(
                        migraphx::literal{migraphx::shape{migraphx::shape::int32_type,
                                                          {static_cast<size_t>(params.batch_size)}},
                                          zeros});
                }

                std::vector<instruction_ref> concat_k_inputs{key, seqlens_k, past_key};
                std::vector<instruction_ref> concat_v_inputs{value, seqlens_k, past_value};

                // Use concat_past_present operator for efficient KV cache concatenation
                present_key = info.add_instruction(
                    make_op("concat_past_present", {{"kv_num_heads", params.num_heads}}),
                    concat_k_inputs);
                present_value = info.add_instruction(
                    make_op("concat_past_present", {{"kv_num_heads", params.num_heads}}),
                    concat_v_inputs);

                key   = present_key.value();
                value = present_value.value();
            }
        }

        if(key->get_shape().dynamic() and value->get_shape().dynamic())
        {
            // Confirm that k/v have correct shape of (batch, num_heads, dyn_past_sl, head_size)
            auto k_shape = key->get_shape();
            auto v_shape = value->get_shape();
            if(k_shape != v_shape)
            {
                MIGRAPHX_THROW("MultiHeadAttention: dynamic key and value must have the same shape");
            }
            if(k_shape.dyn_dims().size() != 4 or k_shape.dyn_dims()[2].is_fixed())
            {
                MIGRAPHX_THROW("MultiHeadAttention: dynamic key and value must have the correct shape");
                //to do: replace error with shape correction
            }
            auto key_out = key;
            auto value_out = value;
            key = info.add_instruction(make_op("fixed_pad"), key);
            value = info.add_instruction(make_op("fixed_pad"), value);
            
            // use dimensions_of to get the dynamic past sequence length as a runtime value
            auto dyn_past_sl = info.add_instruction(make_op("dimensions_of", {{"start", 2}, {"end", 3}}), key_out);
            // add one to get total sequence length
            auto total_seq_len = info.add_instruction(make_op("add"), dyn_past_sl, info.add_literal(migraphx::literal{migraphx::shape{dyn_past_sl->get_shape().type(), {1}}, {1}}));
            total_seq_len = info.add_instruction(make_op("multibroadcast", {{"out_lens", {params.batch_size, params.num_heads}}}), total_seq_len);


            // insert attention op-by-op with greater/where/etc masking pattern of group_query_attention
            auto max_seq_len = key->get_shape().lens()[2];

            auto kt = info.add_instruction(make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), key);
            auto gemm1 = info.add_instruction(make_op("dot"), query, kt);
            
            std::vector<int> range_vec(max_seq_len);
            std::iota(range_vec.begin(), range_vec.end(), 0);
            shape range_s{total_seq_len->get_shape().type(), {max_seq_len}};
            auto range = info.add_literal(range_s, range_vec);
            std::vector<std::size_t> bnsm{static_cast<std::size_t>(params.batch_size), static_cast<std::size_t>(params.num_heads), static_cast<std::size_t>(params.q_sequence_length), static_cast<std::size_t>(max_seq_len)};
            auto bc_range = info.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), range);

            auto scalar_s = shape{query->get_shape().type(), {1}};
            auto ninf = info.add_literal(literal{scalar_s, {-std::numeric_limits<float>::infinity()}});
            ninf = info.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), ninf);
            
            float scale = 1 / std::sqrt(params.head_size);
            if(contains(info.attributes, "scale"))
                scale = parser.parse_value(info.attributes.at("scale")).at<float>();

            if(float_equal(scale, 0.0))
            {
                scale = 1.0f / std::sqrt(static_cast<float>(params.head_size));
            }
            auto scale_ins = info.add_literal(literal{scalar_s, {scale}});
            scale_ins = info.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), scale_ins);
            auto mul = info.add_instruction(make_op("mul"), gemm1, scale_ins);
            
            
            instruction_ref seq_range;
            if(params.q_sequence_length > 1)
            {
                std::vector<int> seq_range_vec(params.q_sequence_length);
                std::iota(seq_range_vec.begin(), seq_range_vec.end(), 0);
                shape seq_range_s{total_seq_len->get_shape().type(), {static_cast<std::size_t>(params.q_sequence_length)}};
                seq_range = info.add_literal(seq_range_s, seq_range_vec);
                seq_range = info.add_instruction(make_op("reshape", {{"dims", {params.q_sequence_length, 1}}}), seq_range);
                seq_range = info.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), seq_range);
                auto causal_mask = info.add_instruction(make_op("greater"), bc_range, seq_range);
                causal_mask = info.add_instruction(make_op("convert", {{"target_type", shape::bool_type}}), causal_mask);
                mul = info.add_instruction(make_op("where"), causal_mask, ninf, mul);
            }

            auto bc_past_sl = info.add_instruction(make_op("reshape", {{"dims", {static_cast<std::size_t>(params.batch_size), static_cast<std::size_t>(params.num_heads), 1, 1}}}), total_seq_len);
            auto mask_comp = info.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), bc_past_sl);
            auto mask = info.add_instruction(make_op("greater"), bc_range, mask_comp);
            mask = info.add_instruction(make_op("convert", {{"target_type", shape::bool_type}}), mask);
            auto where = info.add_instruction(make_op("where"), mask, ninf, mul);
            auto softmax = info.add_instruction(make_op("softmax", {{"axis", 3}}), where);
            auto scores = info.add_instruction(make_op("dot"), softmax, value);
            auto out = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), scores);
            out = info.add_instruction(make_op("reshape", {{"dims", {params.batch_size, params.q_sequence_length, params.hidden_size_v}}}), out);
            return {out, key_out, value_out};

        }

        // Set attention mask and bias when detected on input
        std::optional<instruction_ref> attn_mask;
        if(args.size() > 4)
            attn_mask = create_input_mask(info, args.at(4), query->get_shape(), params);

        std::optional<instruction_ref> attn_bias;
        if(args.size() > 5)
        {
            attn_bias = args.at(5);
        }

        float scale = 1 / std::sqrt(params.head_size);
        if(contains(info.attributes, "scale"))
            scale = parser.parse_value(info.attributes.at("scale")).at<float>();

        auto scale_literal = info.add_literal(
            migraphx::literal{migraphx::shape{query->get_shape().type()}, {scale}});

        auto key_transposed =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), key);

        auto result = info.add_instruction(make_op("dot"), query, key_transposed);

        if(attn_bias.has_value())
        {
            result = info.add_common_op("add", result, attn_bias.value());
        }

        // Apply attention mask
        if(attn_mask.has_value())
        {
            result = info.add_common_op("add", result, attn_mask.value());
        }

        result         = info.add_common_op("mul", result, scale_literal);
        auto qk_output = info.add_instruction(make_op("softmax", {{"axis", -1}}), result);
        // std::cout << "dot2 inputs" << std::endl;
        // result->debug_print();
        // qk_output->debug_print();
        // value->debug_print();
        if(qk_output->get_shape().type() != value->get_shape().type())
        {
            qk_output = info.add_instruction(make_op("convert", {{"target_type", value->get_shape().type()}}), qk_output);
        }
        result         = info.add_instruction(make_op("dot"), qk_output, value);
        result = info.add_instruction(make_op("transpose", {{"permutation", perm}}), result);
        result = info.add_instruction(
            make_op(
                "reshape",
                {{"dims", {params.batch_size, params.q_sequence_length, params.hidden_size_v}}}),
            result);

        // Return outputs based on what's available: present key, present value and qk are optional
        std::vector<instruction_ref> outputs = {result};

        // Add present_key and present_value if past states were provided and non-empty
        if(present_key.has_value() and present_value.has_value())
        {
            outputs.push_back(present_key.value());
            outputs.push_back(present_value.value());
        }

        // Note: QK output could be added here if needed
        // outputs.push_back(qk_output);

        return outputs;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
