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
        if(args.at(1)->get_shape().lens().size() > 1)
        {
            qkv = info.add_instruction(
                make_op("concat", {{"axis", 2}}), args.at(0), args.at(1), args.at(2));
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

        auto q = info.add_instruction(
            make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {num_heads}}}),
            transposed_qkv);
        auto cur_k = info.add_instruction(
            make_op("slice",
                    {{"axes", {1}}, {"starts", {num_heads}}, {"ends", {num_heads + kv_num_heads}}}),
            transposed_qkv);
        auto cur_v = info.add_instruction(
            make_op("slice",
                    {{"axes", {1}},
                     {"starts", {num_heads + kv_num_heads}},
                     {"ends", {num_heads + (2 * kv_num_heads)}}}),
            transposed_qkv);

        if(do_rotary)
        {
            auto seqlens_k = args.at(5);
            auto cos_cache = args.at(7);
            auto sin_cache = args.at(8);

            q = op::builder::add("rotary_embedding",
                                 *info.mod,
                                 {q, seqlens_k, cos_cache, sin_cache},
                                 {{"interleaved", rotary_interleaved}})
                    .at(0);
            cur_k = op::builder::add("rotary_embedding",
                                     *info.mod,
                                     {cur_k, seqlens_k, cos_cache, sin_cache},
                                     {{"interleaved", rotary_interleaved}})
                        .at(0);
        }

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
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
