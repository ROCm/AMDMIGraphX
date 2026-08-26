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
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/dim_ops.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

namespace {

// `sequence_length - 1`, the distance from the last query row back to the first. A known length
// becomes a literal; otherwise it is read from the input at runtime, which folds back to that same
// literal once the length has been specialized.
instruction_ref last_row_offset(const onnx_parser::node_info& info,
                                instruction_ref qkv,
                                const sym::expr& sequence_length,
                                shape::type_t index_type)
{
    const auto value = sym::fixed_value(sequence_length);
    if(value.has_value())
        return info.add_literal(literal{shape{index_type, {1}}, {sym::to<int>(*value) - 1}});
    auto len    = info.add_instruction(make_op("dimensions_of", {{"start", 1}, {"end", 2}}), qkv);
    auto one    = info.add_literal(literal{shape{len->get_shape().type(), {1}}, {1}});
    auto result = info.add_instruction(make_op("sub"), len, one);
    return info.add_instruction(make_op("convert", {{"target_type", index_type}}), result);
}

} // namespace

struct parse_group_query_attention : op_parser<parse_group_query_attention>
{
    std::vector<op_desc> operators() const { return {{"GroupQueryAttention"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        bool do_rotary           = false;
        std::size_t kv_num_heads = 0;
        int local_window_size    = -1;
        std::size_t num_heads    = 0;
        bool rotary_interleaved  = false;
        float scale              = 0.0;
        if(contains(info.attributes, "do_rotary"))
        {
            do_rotary = parser.parse_value(info.attributes.at("do_rotary")).at<bool>();
        }
        if(contains(info.attributes, "kv_num_heads"))
        {
            kv_num_heads = parser.parse_value(info.attributes.at("kv_num_heads")).at<std::size_t>();
        }
        else
        {
            MIGRAPHX_THROW(
                "GroupQueryAttention: Attribute 'kv_num_heads' is required but was not provided.");
        }
        if(contains(info.attributes, "local_window_size"))
        {
            local_window_size =
                parser.parse_value(info.attributes.at("local_window_size")).at<int>();
        }
        if(contains(info.attributes, "num_heads"))
        {
            num_heads = parser.parse_value(info.attributes.at("num_heads")).at<std::size_t>();
        }
        else
        {
            MIGRAPHX_THROW(
                "GroupQueryAttention: Attribute 'num_heads' is required but was not provided.");
        }
        if(contains(info.attributes, "rotary_interleaved"))
        {
            rotary_interleaved =
                parser.parse_value(info.attributes.at("rotary_interleaved")).at<bool>();
        }
        if(contains(info.attributes, "scale"))
        {
            scale = parser.parse_value(info.attributes.at("scale")).at<float>();
        }
        if(contains(info.attributes, "softcap"))
        {
            if(not float_equal(parser.parse_value(info.attributes.at("softcap")).at<float>(), 0.0))
            {
                MIGRAPHX_THROW("GroupQueryAttention: non-zero softcap is not yet supported.");
            }
        }

        if(args.size() < 7 or args.size() > 11)
        {
            MIGRAPHX_THROW("GroupQueryAttention: Wrong number of inputs provided");
        }

        auto qkv = args.at(0);
        if(args.at(1)->get_shape().ndim() > 1)
        {
            qkv = info.add_instruction(
                make_op("concat", {{"axis", 2}}), args.at(0), args.at(1), args.at(2));
        }

        // The sequence length is the only axis that may be symbolic; heads are partitioned out of
        // the hidden dimension while parsing, so every other axis has to be known by then.
        auto q_shape                 = qkv->get_shape();
        const auto sequence_length   = q_shape.sym_dims().at(1);
        const std::size_t batch_size = static_dim(q_shape, 0, "GroupQueryAttention: batch size");
        const std::size_t q_hidden_size =
            static_dim(q_shape, 2, "GroupQueryAttention: hidden size");
        const std::size_t head_size = q_hidden_size / (num_heads + 2 * kv_num_heads);

        const std::vector<sym::expr> bsnh{sym::lit(batch_size),
                                          sequence_length,
                                          sym::lit(num_heads + 2 * kv_num_heads),
                                          sym::lit(head_size)};

        auto transposed_qkv = info.add_instruction(make_reshape(bsnh), qkv);

        transposed_qkv = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}),
                                              transposed_qkv);

        auto qk = info.add_instruction(
            make_op("slice",
                    {{"axes", {1}}, {"starts", {0}}, {"ends", {num_heads + kv_num_heads}}}),
            transposed_qkv);
        auto cur_v = info.add_instruction(make_op("slice",
                                                  {{"axes", {1}},
                                                   {"starts", {num_heads + kv_num_heads}},
                                                   {"ends", {num_heads + (2 * kv_num_heads)}}}),
                                          transposed_qkv);

        auto slk              = args.at(5);
        const auto index_type = slk->get_shape().type();

        // Absolute position of this call's first query token. seqlens_k is the index of the last
        // valid key, so this is zero when the whole prompt arrives at once and the past length
        // when a single token is appended. Deriving position from it rather than branching on
        // sequence_length lets one graph serve both cases.
        auto first_pos =
            info.add_common_op("sub", slk, last_row_offset(info, qkv, sequence_length, index_type));

        if(do_rotary)
        {
            qk = op::builder::add("rotary_embedding",
                                  *info.mod,
                                  {qk, first_pos, args.at(7), args.at(8)},
                                  {{"interleaved", rotary_interleaved}})
                     .at(0);
        }

        auto q = info.add_instruction(
            make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {num_heads}}}), qk);
        auto cur_k = info.add_instruction(
            make_op("slice",
                    {{"axes", {1}}, {"starts", {num_heads}}, {"ends", {num_heads + kv_num_heads}}}),
            qk);

        auto k = args.at(3);
        auto v = args.at(4);
        std::vector<instruction_ref> concat_k_inputs{cur_k, slk, k};
        std::vector<instruction_ref> concat_v_inputs{cur_v, slk, v};

        k = info.add_instruction(make_op("concat_past_present", {{"kv_num_heads", kv_num_heads}}),
                                 concat_k_inputs);
        v = info.add_instruction(make_op("concat_past_present", {{"kv_num_heads", kv_num_heads}}),
                                 concat_v_inputs);

        auto k_out = k;
        auto v_out = v;

        auto kv_num_heads_factor = num_heads / kv_num_heads;
        auto max_seq_len         = k->get_shape().lens()[2];

        if(kv_num_heads_factor != 1)
        {
            auto kv_new_lens  = k->get_shape().lens();
            kv_new_lens.at(1) = num_heads;
            k                 = info.add_instruction(make_op("unsqueeze", {{"axes", {2}}}), k);
            v                 = info.add_instruction(make_op("unsqueeze", {{"axes", {2}}}), v);
            auto kv_unsqueezed_lens  = k->get_shape().lens();
            kv_unsqueezed_lens.at(2) = kv_num_heads_factor;
            k = info.add_instruction(make_op("multibroadcast", {{"out_lens", kv_unsqueezed_lens}}),
                                     k);
            v = info.add_instruction(make_op("multibroadcast", {{"out_lens", kv_unsqueezed_lens}}),
                                     v);
            k = info.add_instruction(make_op("reshape", {{"dims", kv_new_lens}}), k);
            v = info.add_instruction(make_op("reshape", {{"dims", kv_new_lens}}), v);
        }
        auto kt    = info.add_instruction(make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);
        auto gemm1 = info.add_instruction(make_op("dot"), q, kt);

        std::vector<int> range_vec(max_seq_len);
        std::iota(range_vec.begin(), range_vec.end(), 0);
        shape range_s{index_type, {max_seq_len}};
        auto range = info.add_literal(range_s, range_vec);
        const std::vector<sym::expr> bnsm{
            sym::lit(batch_size), sym::lit(num_heads), sequence_length, sym::lit(max_seq_len)};
        auto bc_range = info.add_instruction(make_multibroadcast(bnsm), range);

        auto scalar_s = shape{transposed_qkv->get_shape().type(), {1}};
        auto ninf = info.add_literal(literal{scalar_s, {-std::numeric_limits<float>::infinity()}});
        ninf      = info.add_instruction(make_multibroadcast(bnsm), ninf);

        if(float_equal(scale, 0.0))
        {
            scale = 1.0f / std::sqrt(static_cast<float>(head_size));
        }
        auto scale_ins = info.add_literal(literal{scalar_s, {scale}});
        scale_ins      = info.add_instruction(make_multibroadcast(bnsm), scale_ins);
        auto mul       = info.add_instruction(make_op("mul"), gemm1, scale_ins);

        // Absolute cache position of each query row. Shifting this call's rows by first_pos makes
        // the causal mask correct for a fresh prompt and for a continuation alike, and it also
        // subsumes the padding mask: every row index is at most seqlens_k, so masking keys beyond
        // the row masks everything that masking keys beyond seqlens_k would have.
        // Broadcast the offset the same way the padding mask used to, so that a single-token step
        // still collapses to the same cheap per-batch scalar read inside the fused attention
        // kernel once the all-zero range below folds away.
        auto row_pos = info.add_instruction(
            make_op("multibroadcast", {{"out_lens", {batch_size, num_heads}}}), first_pos);
        row_pos = info.add_instruction(
            make_op("reshape", {{"dims", {batch_size, num_heads, 1, 1}}}), row_pos);
        row_pos = info.add_instruction(make_multibroadcast(bnsm), row_pos);

        auto seq_range = insert_iota(*info.mod,
                                     info.mod->end(),
                                     {sym::lit(1), sym::lit(1), sequence_length, sym::lit(1)},
                                     2,
                                     qkv,
                                     1,
                                     index_type);
        seq_range      = info.add_instruction(make_multibroadcast(bnsm), seq_range);
        row_pos        = info.add_instruction(make_op("add"), row_pos, seq_range);

        if(local_window_size > 0)
        {
            // The two phases disagree by one key on where the window starts. Preserved verbatim
            // until the intended bound is confirmed against onnxruntime; only the row index it is
            // measured from has been unified here. Until that is settled the bound cannot be
            // written without knowing which phase it is for.
            const auto seq_value = sym::fixed_value(sequence_length);
            if(not seq_value.has_value())
                MIGRAPHX_THROW("GroupQueryAttention: local_window_size is not supported with a "
                               "symbolic sequence length");
            bool is_prompt       = sym::to<std::size_t>(*seq_value) > 1;
            auto window_size_lit = info.add_literal(
                literal{shape{index_type, {1}},
                        {is_prompt ? -local_window_size : -(local_window_size + 1)}});
            window_size_lit  = info.add_instruction(make_multibroadcast(bnsm), window_size_lit);
            auto window_comp = info.add_instruction(make_op("add"), row_pos, window_size_lit);
            auto window_mask = info.add_instruction(make_op("greater"), window_comp, bc_range);
            window_mask      = info.add_instruction(
                make_op("convert", {{"target_type", shape::bool_type}}), window_mask);
            mul = info.add_instruction(make_op("where"), window_mask, ninf, mul);
        }
        auto mask = info.add_instruction(make_op("greater"), bc_range, row_pos);
        mask = info.add_instruction(make_op("convert", {{"target_type", shape::bool_type}}), mask);
        auto where   = info.add_instruction(make_op("where"), mask, ninf, mul);
        auto softmax = info.add_instruction(make_op("softmax", {{"axis", 3}}), where);
        auto scores  = info.add_instruction(make_op("dot"), softmax, v);
        auto out =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), scores);
        out = info.add_instruction(
            make_reshape({sym::lit(batch_size), sequence_length, sym::lit(head_size * num_heads)}),
            out);

        return {out, k_out, v_out};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
