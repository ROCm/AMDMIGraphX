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
        std::size_t num_heads    = 1;
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
        if(contains(info.attributes, "local_window_size"))
        {
            local_window_size =
                parser.parse_value(info.attributes.at("local_window_size")).at<int>();
        }
        if(contains(info.attributes, "num_heads"))
        {
            num_heads = parser.parse_value(info.attributes.at("num_heads")).at<std::size_t>();
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

        if(args.size() < 7 or args.size() > 9)
        {
            MIGRAPHX_THROW("GroupQueryAttention: Wrong number of inputs provided");
        }

        auto new_args = args;
        if(args.at(1)->get_shape().lens().size() > 1)
        {
            new_args[0] = info.add_instruction(
                make_op("concat", {{"axis", 2}}), args.at(0), args.at(1), args.at(2));
        }

        auto inputs = args;

        auto q_shape                      = inputs[0]->get_shape();
        const auto& q_lens                = q_shape.lens();
        const std::size_t batch_size      = q_lens[0];
        const std::size_t sequence_length = q_lens[1];
        std::size_t q_hidden_size         = q_lens[2];
        std::size_t head_size             = q_hidden_size / (num_heads + 2 * kv_num_heads);

        std::vector<std::size_t> bsnh{
            batch_size, sequence_length, num_heads + 2 * kv_num_heads, head_size};

        auto transposed_qkv = info.add_instruction(make_op("reshape", {{"dims", bsnh}}), inputs.at(0));

        transposed_qkv = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), transposed_qkv);

        auto rotary_qkv = transposed_qkv;
        if(do_rotary)
        {
            std::vector<instruction_ref> rotary_inputs{
                transposed_qkv, inputs.at(5), inputs.at(7), inputs.at(8)};
            rotary_qkv =
                info.add_instruction(
                    make_op("gqa_rotary_embedding", {{"kv_num_heads", kv_num_heads}, {"num_heads", num_heads}, {"rotary_interleaved", rotary_interleaved}}), 
                                                    rotary_inputs);
        }

        auto pres_k   = inputs.at(3);
        auto pres_v   = inputs.at(4);
        auto slk      = inputs.at(5);
        auto rotary_k = info.add_instruction(
            make_op("slice",
                    {{"axes", {1}}, {"starts", {num_heads}}, {"ends", {num_heads + kv_num_heads}}}),
            rotary_qkv);
        auto rotary_v = info.add_instruction(
            make_op("slice",
                    {{"axes", {1}},
                    {"starts", {num_heads + kv_num_heads}},
                    {"ends", {num_heads + (2 * kv_num_heads)}}}),
            rotary_qkv);
        std::vector<instruction_ref> concat_k_inputs{rotary_k, slk, pres_k};
        std::vector<instruction_ref> concat_v_inputs{rotary_v, slk, pres_v};

        pres_k = info.add_instruction(
            make_op("concat_past_present", {{"kv_num_heads", kv_num_heads}, {"num_heads", num_heads}}),
            concat_k_inputs);
        pres_v = info.add_instruction(
            make_op("concat_past_present", {{"kv_num_heads", kv_num_heads}, {"num_heads", num_heads}}),
            concat_v_inputs);

        // Adding 1 to seq_lens_k, aka past_seq_lens, to allow range literals to start at 0.
        // Putting the add inside the mlir module currently causes an error on their side,
        // so we're leaving it here until that can be solved.
        auto one_lit = info.add_literal(literal{shape{inputs.at(5)->get_shape().type(), {1}}, {1}});
        one_lit = info.add_instruction(
            make_op("multibroadcast", {{"out_lens", inputs.at(5)->get_shape().lens()}}),
            one_lit);
        auto total_sl =
            info.add_instruction(make_op("add"), inputs.at(5), one_lit);

        // auto get_tuple_elm_0 = std::next(ins);
        // auto get_tuple_elm_1 = std::next(get_tuple_elm_0);
        // auto get_tuple_elm_2 = std::next(get_tuple_elm_1);

        // mpm.get_module().replace_instruction(get_tuple_elm_2, pres_v);
        // mpm.get_module().replace_instruction(get_tuple_elm_1, pres_k);

        auto kv_num_heads_factor = num_heads / kv_num_heads;
        auto max_seq_len         = pres_k->get_shape().lens()[2];
        total_sl                 = info.add_instruction(make_op("multibroadcast", {{"out_lens", {batch_size, num_heads}}}), total_sl);
        // std::vector<instruction_ref> new_inputs{rotary_qkv, pres_k, pres_v, total_sl};

        // module m_attn;
        // std::vector<instruction_ref> attn_inputs = {rotary_qkv, pres_k, pres_v, total_sl};
        // std::unordered_map<instruction_ref, instruction_ref> map_main_to_mattn;
        // m_attn.add_params(attn_inputs, &map_main_to_mattn);

        auto q = info.add_instruction(
            make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {num_heads}}}),
            rotary_qkv);
        auto k = pres_k;
        auto v = pres_v;
        if(kv_num_heads_factor != 1)
        {
            auto kv_new_lens  = k->get_shape().lens();
            kv_new_lens.at(1) = num_heads;
            k                 = info.add_instruction(make_op("unsqueeze", {{"axes", {2}}}), k);
            v                 = info.add_instruction(make_op("unsqueeze", {{"axes", {2}}}), v);
            auto kv_unsqueezed_lens  = k->get_shape().lens();
            kv_unsqueezed_lens.at(2) = kv_num_heads_factor;
            k                        = info.add_instruction(
                make_op("multibroadcast", {{"out_lens", kv_unsqueezed_lens}}), k);
            v = info.add_instruction(
                make_op("multibroadcast", {{"out_lens", kv_unsqueezed_lens}}), v);
            k = info.add_instruction(make_op("reshape", {{"dims", kv_new_lens}}), k);
            v = info.add_instruction(make_op("reshape", {{"dims", kv_new_lens}}), v);
        }
        auto kt = info.add_instruction(make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);
        auto gemm1 = info.add_instruction(make_op("dot"), q, kt);

        std::vector<int> range_vec(max_seq_len);
        std::iota(range_vec.begin(), range_vec.end(), 0);
        shape range_s{total_sl->get_shape().type(), {max_seq_len}};
        auto range = info.add_literal(range_s, range_vec);
        std::vector<std::size_t> bnsm{batch_size, num_heads, sequence_length, max_seq_len};
        auto bc_range =
            info.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), range);

        auto scalar_s = shape{rotary_qkv->get_shape().type(), {1}};
        auto ninf =
            info.add_literal(literal{scalar_s, {-std::numeric_limits<float>::infinity()}});
        ninf = info.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), ninf);

        if(float_equal(scale, 0.0))
        {
            scale = 1.0f / std::sqrt(static_cast<float>(head_size));
        }
        auto scale_ins = info.add_literal(literal{scalar_s, {scale}});
        scale_ins =
            info.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), scale_ins);
        auto mul = info.add_instruction(make_op("mul"), gemm1, scale_ins);

        if(sequence_length > 1)
        {
            std::vector<int> seq_range_vec(sequence_length);
            std::iota(seq_range_vec.begin(), seq_range_vec.end(), 0);
            shape seq_range_s{total_sl->get_shape().type(), {sequence_length}};
            auto seq_range = info.add_literal(seq_range_s, seq_range_vec);
            seq_range = info.add_instruction(make_op("reshape", {{"dims", {sequence_length, 1}}}),
                                            seq_range);
            seq_range =
                info.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), seq_range);
            auto causal_mask = info.add_instruction(make_op("greater"), bc_range, seq_range);
            causal_mask      = info.add_instruction(
                make_op("convert", {{"target_type", shape::bool_type}}), causal_mask);
            mul = info.add_instruction(make_op("where"), causal_mask, ninf, mul);
        }

        auto bc_total_sl =
            info.add_instruction(make_op("reshape", {{"dims", {batch_size, num_heads, 1, 1}}}),
                                total_sl);
        auto mask_comp =
            info.add_instruction(make_op("multibroadcast", {{"out_lens", bnsm}}), bc_total_sl);
        auto mask = info.add_instruction(make_op("greater"), bc_range, mask_comp);
        mask =
            info.add_instruction(make_op("convert", {{"target_type", shape::bool_type}}), mask);
        auto where   = info.add_instruction(make_op("where"), mask, ninf, mul);
        auto softmax = info.add_instruction(make_op("softmax", {{"axis", 3}}), where);
        auto scores  = info.add_instruction(make_op("dot"), softmax, v);
        auto out =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), scores);
        out = info.add_instruction(
            make_op("reshape", {{"dims", {batch_size, sequence_length, head_size * num_heads}}}), out);
        
        return {out, pres_k, pres_v};

        // auto gqa             = info.add_instruction(make_op("group_query_attention",
        //                                                     {{"do_rotary", do_rotary},
        //                                                      {"kv_num_heads", kv_num_heads},
        //                                                      {"local_window_size", local_window_size},
        //                                                      {"num_heads", num_heads},
        //                                                      {"rotary_interleaved", rotary_interleaved},
        //                                                      {"scale", scale}}),
        //                                 new_args);
        // auto gqa_output      = info.add_instruction(make_op("get_tuple_elem", {{"index", 0}}), gqa);
        // auto gqa_present_key = info.add_instruction(make_op("get_tuple_elem", {{"index", 1}}), gqa);
        // auto gqa_present_value =
        //     info.add_instruction(make_op("get_tuple_elem", {{"index", 2}}), gqa);

        // return {gqa_output, gqa_present_key, gqa_present_value};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
