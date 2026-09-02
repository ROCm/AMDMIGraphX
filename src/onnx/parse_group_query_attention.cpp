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
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_group_query_attention : op_parser<parse_group_query_attention>
{
    std::vector<op_desc> operators() const { return {{"GroupQueryAttention"}}; }

    struct gqa_options
    {
        bool do_rotary           = false;
        std::size_t kv_num_heads = 0;
        int local_window_size    = -1;
        std::size_t num_heads    = 0;
        bool rotary_interleaved  = false;
        float scale              = 0.0;
    };

    static instruction_ref insert_rotary(module& m,
                                         bool interleaved,
                                         std::size_t sequence_length,
                                         std::vector<instruction_ref> args)
    {
        // GQA position semantics: prefill starts from 0, decode uses seqlens_k
        auto& pos_ids = args.at(1);
        if(sequence_length > 1)
        {
            pos_ids = m.add_literal(literal{shape{pos_ids->get_shape().type(), {1}}, {0}});
        }
        return op::builder::add("rotary_embedding", m, args, {{"interleaved", interleaved}}).at(0);
    }

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

        if(qkv->get_shape().symbolic())
        {
            return parse_symbolic(
                info,
                args,
                qkv,
                {do_rotary, kv_num_heads, local_window_size, num_heads, rotary_interleaved, scale});
        }

        auto q_shape                      = qkv->get_shape();
        const auto& q_lens                = q_shape.lens();
        const std::size_t batch_size      = q_lens[0];
        const std::size_t sequence_length = q_lens[1];
        std::size_t q_hidden_size         = q_lens[2];
        std::size_t head_size             = q_hidden_size / (num_heads + 2 * kv_num_heads);

        std::vector<std::size_t> bsnh{
            batch_size, sequence_length, num_heads + 2 * kv_num_heads, head_size};

        auto transposed_qkv = info.add_instruction(make_op("reshape", {{"dims", bsnh}}), qkv);

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

        if(do_rotary)
        {
            qk = insert_rotary(*info.mod,
                               rotary_interleaved,
                               sequence_length,
                               {qk, args.at(5), args.at(7), args.at(8)});
        }

        auto q = info.add_instruction(
            make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {num_heads}}}), qk);
        auto cur_k = info.add_instruction(
            make_op("slice",
                    {{"axes", {1}}, {"starts", {num_heads}}, {"ends", {num_heads + kv_num_heads}}}),
            qk);

        auto k   = args.at(3);
        auto v   = args.at(4);
        auto slk = args.at(5);
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
        auto past_sl             = info.add_instruction(
            make_op("multibroadcast", {{"out_lens", {batch_size, num_heads}}}), slk);

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
        shape range_s{past_sl->get_shape().type(), {max_seq_len}};
        auto range = info.add_literal(range_s, range_vec);
        std::vector<std::size_t> bnsm{batch_size, num_heads, sequence_length, max_seq_len};
        auto bc_range =
            info.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), range);

        auto scalar_s = shape{transposed_qkv->get_shape().type(), {1}};
        auto ninf = info.add_literal(literal{scalar_s, {-std::numeric_limits<float>::infinity()}});
        ninf      = info.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), ninf);

        if(float_equal(scale, 0.0))
        {
            scale = 1.0f / std::sqrt(static_cast<float>(head_size));
        }
        auto scale_ins = info.add_literal(literal{scalar_s, {scale}});
        scale_ins =
            info.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), scale_ins);
        auto mul = info.add_instruction(make_op("mul"), gemm1, scale_ins);

        instruction_ref seq_range;
        if(sequence_length > 1)
        {
            std::vector<int> seq_range_vec(sequence_length);
            std::iota(seq_range_vec.begin(), seq_range_vec.end(), 0);
            shape seq_range_s{past_sl->get_shape().type(), {sequence_length}};
            seq_range = info.add_literal(seq_range_s, seq_range_vec);
            seq_range = info.add_instruction(make_op("reshape", {{"dims", {sequence_length, 1}}}),
                                             seq_range);
            seq_range =
                info.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), seq_range);
            auto causal_mask = info.add_instruction(make_op("greater"), bc_range, seq_range);
            causal_mask      = info.add_instruction(
                make_op("convert", {{"target_type", shape::bool_type}}), causal_mask);
            mul = info.add_instruction(make_op("where"), causal_mask, ninf, mul);
        }

        auto bc_past_sl = info.add_instruction(
            make_op("reshape", {{"dims", {batch_size, num_heads, 1, 1}}}), past_sl);
        auto mask_comp =
            info.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), bc_past_sl);
        if(local_window_size > 0)
        {
            bool is_prompt       = sequence_length > 1;
            auto window_size_lit = info.add_literal(
                migraphx::literal{migraphx::shape{past_sl->get_shape().type(), {1}},
                                  {is_prompt ? -local_window_size : -(local_window_size + 1)}});
            window_size_lit = info.add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", bnsm}}), window_size_lit);
            auto window_comp = info.add_instruction(
                migraphx::make_op("add"), is_prompt ? seq_range : mask_comp, window_size_lit);
            auto window_mask =
                info.add_instruction(migraphx::make_op("greater"), window_comp, bc_range);
            window_mask = info.add_instruction(
                migraphx::make_op("convert", {{"target_type", migraphx::shape::bool_type}}),
                window_mask);
            mul = info.add_instruction(migraphx::make_op("where"), window_mask, ninf, mul);
        }
        auto mask = info.add_instruction(make_op("greater"), bc_range, mask_comp);
        mask = info.add_instruction(make_op("convert", {{"target_type", shape::bool_type}}), mask);
        auto where   = info.add_instruction(make_op("where"), mask, ninf, mul);
        auto softmax = info.add_instruction(make_op("softmax", {{"axis", 3}}), where);
        auto scores  = info.add_instruction(make_op("dot"), softmax, v);
        auto out =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), scores);
        out = info.add_instruction(
            make_op("reshape", {{"dims", {batch_size, sequence_length, head_size * num_heads}}}),
            out);

        return {out, k_out, v_out};
    }

    static std::size_t fixed_dim(const shape::dynamic_dimension& dd, const std::string& what)
    {
        if(not dd.is_fixed())
            MIGRAPHX_THROW("GroupQueryAttention: " + what + " must be a fixed dimension");
        return dd.get_interval().min;
    }

    // Symbolic path: batch and sequence dimensions are symbolic expressions, so every
    // parse-time prompt/decode branch of the static path is replaced with the unified
    // run-time form. Per-token key positions are seqlens_k + 1 - seq + i, which reduces
    // to 0..seq-1 for a prompt (seqlens_k = seq - 1) and to seqlens_k for decode, so a
    // single causal mask j > position covers both cases.
    static std::vector<instruction_ref> parse_symbolic(const onnx_parser::node_info& info,
                                                       const std::vector<instruction_ref>& args,
                                                       instruction_ref qkv,
                                                       const gqa_options& opts)
    {
        const auto qkv_shape = qkv->get_shape();
        if(qkv_shape.ndim() != 3)
            MIGRAPHX_THROW("GroupQueryAttention: expected packed qkv of rank 3");
        const auto& qkv_dims     = qkv_shape.dyn_dims();
        const auto total_heads   = static_cast<int64_t>(opts.num_heads + 2 * opts.kv_num_heads);
        const auto q_hidden_size = fixed_dim(qkv_dims[2], "hidden size");
        if(q_hidden_size % total_heads != 0)
            MIGRAPHX_THROW("GroupQueryAttention: hidden size is not divisible by the number of "
                           "heads");
        const auto head_size = static_cast<int64_t>(q_hidden_size / total_heads);
        const auto num_heads = static_cast<int64_t>(opts.num_heads);
        const auto kv_heads  = static_cast<int64_t>(opts.kv_num_heads);

        auto transposed_qkv = info.add_instruction(
            make_op("reshape", {{"dims", {0, -1, total_heads, head_size}}}), qkv);
        transposed_qkv = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}),
                                              transposed_qkv);

        auto qk = info.add_instruction(
            make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {num_heads + kv_heads}}}),
            transposed_qkv);
        auto cur_v = info.add_instruction(
            make_op("slice",
                    {{"axes", {1}}, {"starts", {num_heads + kv_heads}}, {"ends", {total_heads}}}),
            transposed_qkv);

        // Per-token key positions {batch, seq, 1}: seqlens_k + 1 - seq + i
        auto slk   = args.at(5);
        auto slk64 = slk;
        if(slk64->get_shape().type() != shape::int64_type)
            slk64 = info.add_instruction(make_op("convert", {{"target_type", shape::int64_type}}),
                                         slk64);
        if(slk64->get_shape().ndim() != 1)
            slk64 = info.add_instruction(make_op("reshape", {{"dims", {-1}}}), slk64);
        auto seq_rt =
            info.add_instruction(make_op("dimensions_of", {{"start", 1}, {"end", 2}}), qkv);
        auto zero     = info.add_literal(literal{shape{shape::int64_type, {1}}, {0}});
        auto one      = info.add_literal(literal{shape{shape::int64_type, {1}}, {1}});
        auto past_len = info.add_common_op("sub", slk64, seq_rt);
        past_len      = info.add_common_op("add", past_len, one);
        auto iota     = info.add_instruction(
            make_op("dynamic_range", {{"output_dim", to_value(qkv_dims[1])}}), zero, seq_rt, one);
        auto past_len_b = info.add_instruction(make_op("unsqueeze", {{"axes", {1, 2}}}), past_len);
        auto iota_b     = info.add_instruction(make_op("unsqueeze", {{"axes", {0, 2}}}), iota);
        auto positions  = info.add_common_op("add", past_len_b, iota_b);

        if(opts.do_rotary)
        {
            qk = op::builder::add("rotary_embedding",
                                  *info.mod,
                                  {qk, positions, args.at(7), args.at(8)},
                                  {{"interleaved", opts.rotary_interleaved}})
                     .at(0);
        }

        auto q = info.add_instruction(
            make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {num_heads}}}), qk);
        auto cur_k = info.add_instruction(
            make_op("slice",
                    {{"axes", {1}}, {"starts", {num_heads}}, {"ends", {num_heads + kv_heads}}}),
            qk);

        auto k = info.add_instruction(
            make_op("concat_past_present", {{"kv_num_heads", opts.kv_num_heads}}),
            cur_k,
            slk,
            args.at(3));
        auto v = info.add_instruction(
            make_op("concat_past_present", {{"kv_num_heads", opts.kv_num_heads}}),
            cur_v,
            slk,
            args.at(4));

        auto k_out = k;
        auto v_out = v;
        if(k->get_shape().dynamic())
            MIGRAPHX_THROW("GroupQueryAttention: past kv-cache must have a static shape");

        // The cache is static, so repeating the kv heads for grouped queries stays on the
        // static path.
        const auto kv_num_heads_factor = opts.num_heads / opts.kv_num_heads;
        const auto max_seq_len         = k->get_shape().lens()[2];
        if(kv_num_heads_factor != 1)
        {
            auto kv_new_lens  = k->get_shape().lens();
            kv_new_lens.at(1) = opts.num_heads;
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
        auto kt = info.add_instruction(make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);

        // Reuse the dimension expressions of the symbolic operand so dot sees matching
        // symbols; the static kv operand broadcasts over the symbolic batch dimension.
        auto q_dims        = q->get_shape().dyn_dims();
        auto bcast_for_dot = [&](instruction_ref static_ins) {
            auto dims = static_ins->get_shape().to_symbolic().dyn_dims();
            dims[0]   = q_dims[0];
            return info.add_instruction(
                make_op("multibroadcast", {{"out_dyn_dims", to_value(dims)}}), static_ins);
        };
        auto gemm1 = info.add_instruction(make_op("dot"), q, bcast_for_dot(kt));

        auto scale = opts.scale;
        if(float_equal(scale, 0.0))
            scale = 1.0f / std::sqrt(static_cast<float>(head_size));
        const auto dtype = qkv_shape.type();
        auto scale_lit   = info.add_literal(literal{shape{dtype, {1}}, {scale}});
        auto scaled      = info.add_common_op("mul", gemm1, scale_lit);

        // Causal mask: key position j is masked when j > position of the query token.
        std::vector<int64_t> col_vec(max_seq_len);
        std::iota(col_vec.begin(), col_vec.end(), 0);
        auto col =
            info.add_literal(literal{shape{shape::int64_type, {1, 1, 1, max_seq_len}}, col_vec});
        auto positions_b = info.add_instruction(make_op("unsqueeze", {{"axes", {1}}}), positions);
        auto mask        = info.add_common_op("greater", col, positions_b);
        if(opts.local_window_size > 0)
        {
            // Keys older than the window are masked: j < position - window_size.
            auto window_lit =
                info.add_literal(literal{shape{shape::int64_type, {1}}, {opts.local_window_size}});
            auto window_comp = info.add_common_op("sub", positions_b, window_lit);
            auto window_mask = info.add_common_op("greater", window_comp, col);
            mask             = info.add_common_op("logical_or", mask, window_mask);
        }
        mask = info.add_instruction(make_op("convert", {{"target_type", shape::bool_type}}), mask);

        auto out_dims = scaled->get_shape().dyn_dims();
        mask          = info.add_instruction(
            make_op("multibroadcast", {{"out_dyn_dims", to_value(out_dims)}}), mask);
        auto ninf =
            info.add_literal(literal{shape{dtype, {1}}, {-std::numeric_limits<float>::infinity()}});
        ninf = info.add_instruction(
            make_op("multibroadcast", {{"out_dyn_dims", to_value(out_dims)}}), ninf);
        auto where   = info.add_instruction(make_op("where"), mask, ninf, scaled);
        auto softmax = info.add_instruction(make_op("softmax", {{"axis", 3}}), where);
        auto scores  = info.add_instruction(make_op("dot"), softmax, bcast_for_dot(v));
        auto out =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), scores);
        out = info.add_instruction(make_op("reshape", {{"dims", {0, -1, num_heads * head_size}}}),
                                   out);

        return {out, k_out, v_out};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
