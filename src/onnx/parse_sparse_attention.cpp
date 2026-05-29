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

#include "migraphx/instruction_ref.hpp"
#include "migraphx/onnx/onnx_parser.hpp"
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_sparse_attention : op_parser<parse_sparse_attention>
{
    struct sparse_attn_parameters
    {
        size_t num_heads;
        size_t kv_num_heads;
        size_t sparse_block_size;
        size_t batch_size;
        size_t sequence_length;
        size_t hidden_size;
        size_t head_size;
        size_t max_cache_sequence_length;
        size_t num_layouts;
        size_t max_blocks;
        size_t max_nnz;
        size_t max_sequence_length;
        size_t max_rotary_sequence_length;
        size_t rotary_embed_dim;
        float scale;
        bool is_packed_qkv;
        bool do_rotary;
        bool rotary_interleaved;
    };

    std::vector<op_desc> operators() const { return {{"SparseAttention"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        auto params = make_params(parser, info, args);
        verify_attributes(params);
        verify_inputs(params, args);

        /* OP IMPL*/
        // Reshape qkv
        // Split up qkv into q, k, v
        // Do past_present_concat
        // Do GQA if necessary
        // Attention probabilities
        // Masked softmax
        // Attention scores
        /* OP IMPL*/
    }

    template <typename T>
    T parse_attribute(const onnx_parser::node_info& info,
                      const onnx_parser& parser,
                      const std::string& name,
                      bool required,
                      T default_value = T{}) const
    {
        if(not contains(info.attributes, name))
        {
            if(required)
                MIGRAPHX_THROW("SparseAttention: " + name + " attribute is required");

            return default_value;
        }

        return parser.parse_value(info.attributes.at(name)).at<T>();
    }

    sparse_attn_parameters make_params(const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        sparse_attn_parameters params;

        params.num_heads         = parse_attribute<int>(info, parser, "num_heads", true);
        params.kv_num_heads      = parse_attribute<int>(info, parser, "kv_num_heads", true);
        params.sparse_block_size = parse_attribute<int>(info, parser, "sparse_block_size", true);
        params.scale             = parse_attribute(info, parser, "scale", false, 0.0f);
        params.do_rotary         = parse_attribute(info, parser, "do_rotary", false, false);
        params.rotary_interleaved =
            parse_attribute(info, parser, "rotary_interleaved", false, false);

        auto num_args          = args.size();
        auto expected_num_args = params.do_rotary ? 11 : 9;
        if(num_args != expected_num_args)
            MIGRAPHX_THROW("SparseAttention: when do_rotary=" +
                           std::string{params.do_rotary ? "true" : "false"} +
                           ", number of inputs should be " + std::to_string(expected_num_args) +
                           ", actual: " + std::to_string(num_args));

        if(args[1]->is_undefined() xor args[2]->is_undefined())
        {
            MIGRAPHX_THROW("SparseAttention: 2nd and 3rd inputs(k and v) must both be defined or "
                           "both be undefined");
        }

        const auto arg_shapes = to_shapes(args);
        // TODO handle both packed qkv and unpacked qkv
        params.is_packed_qkv             = args[1]->is_undefined();
        params.batch_size                = arg_shapes[0].lens()[0];
        params.sequence_length           = arg_shapes[0].lens()[1];
        params.hidden_size               = arg_shapes[0].lens()[2];
        auto total_num_heads             = params.num_heads + 2 * params.kv_num_heads;
        params.head_size                 = params.hidden_size / total_num_heads;
        params.max_cache_sequence_length = arg_shapes[3].lens()[2];
        params.num_layouts               = arg_shapes[5].lens()[0];
        params.max_blocks                = arg_shapes[5].lens()[1] - 1;
        params.max_sequence_length       = params.max_blocks * params.sparse_block_size;
        params.max_nnz                   = arg_shapes[6].lens()[1];

        if(params.do_rotary)
        {
            params.max_rotary_sequence_length = arg_shapes[9].lens()[0];
            params.rotary_embed_dim           = arg_shapes[9].lens()[1] * 2;
        }

        return params;
    }

    void verify_attributes(const sparse_attn_parameters& params) const
    {
        if(params.num_heads <= 0)
            MIGRAPHX_THROW("SparseAttention: num_heads must be greater than zero, actual: " +
                           std::to_string(params.num_heads));
        if(params.kv_num_heads <= 0)
            MIGRAPHX_THROW("SparseAttention: kv_num_heads must be greater than zero, actual: " +
                           std::to_string(params.kv_num_heads));
        if(params.sparse_block_size <= 0)
            MIGRAPHX_THROW(
                "SparseAttention: sparse_block_size must be greater than zero, actual: " +
                std::to_string(params.sparse_block_size));
    }

    void verify_inputs(const sparse_attn_parameters& params,
                       const std::vector<instruction_ref>& args) const
    {
        const auto qkv                        = args[0];
        const auto k                          = args[1];
        const auto v                          = args[2];
        const auto past_k                     = args[3];
        const auto past_v                     = args[4];
        const auto block_row_indices          = args[5];
        const auto block_col_indices          = args[6];
        const auto total_sequence_length      = args[7];
        const auto key_total_sequence_lengths = args[8];

        if(not params.is_packed_qkv)
            MIGRAPHX_THROW("SparseAttention: only packed qkv supported, key and value inputs must "
                           "be undefined");
        if(qkv->get_shape().ndim() != 3)
            MIGRAPHX_THROW("SparseAttention: query input rank must be 3, actual: " +
                           std::to_string(qkv->get_shape().ndim()));

        if((params.hidden_size % (params.num_heads + 2 * params.kv_num_heads)) != 0)
            MIGRAPHX_THROW(
                "SparseAttention: QKV hidden size must be divisible by (num_heads + 2 * "
                "kv_num_heads), actual hidden size: " +
                std::to_string(params.hidden_size) + ", actual num_heads and kv_num_heads: " +
                std::to_string(params.num_heads) + ", " + std::to_string(params.kv_num_heads));
        auto head_size_factor = params.do_rotary ? 16 : 8;
        if((params.head_size % head_size_factor) != 0)
            MIGRAPHX_THROW("SparseAttention: when do_rotary=" +
                           std::string{params.do_rotary ? "true" : "false"} +
                           ", head_size must be a multiple of " + std::to_string(head_size_factor) +
                           ", actual: " + std::to_string(params.head_size));

        auto&& past_k_lens = past_k->get_shape().lens();
        if(past_k_lens.size() != 4)
            MIGRAPHX_THROW("SparseAttention: past_key rank should be 4, actual: " +
                           std::to_string(past_k_lens.size()));
        if(past_k_lens[0] != params.batch_size)
            MIGRAPHX_THROW("SparseAttention: past_key input dim 0 must be equal to batch_size: " +
                           std::to_string(params.batch_size) +
                           ", actual: " + std::to_string(past_k_lens[0]));
        if(past_k_lens[1] != params.kv_num_heads)
            MIGRAPHX_THROW("SparseAttention: past_key input dim 1 must be equal to kv_num_heads: " +
                           std::to_string(params.kv_num_heads) +
                           ", actual: " + std::to_string(past_k_lens[1]));
        if(past_k_lens[3] != params.head_size)
            MIGRAPHX_THROW("SparseAttention: past_key input dim 3 must be equal to head_size: " +
                           std::to_string(params.head_size) +
                           ", actual: " + std::to_string(past_k_lens[3]));
        if(past_k->get_shape() != past_v->get_shape())
        {
            std::stringstream err_msg;
            err_msg << "SparseAttention: past_key and past_value inputs must have the same "
                       "shape, actual shapes for past key and past value: ";
            err_msg << past_k->get_shape() << ", " << past_v->get_shape();
            MIGRAPHX_THROW(err_msg.str());
        }

        auto&& block_row_indices_lens = block_row_indices->get_shape().lens();
        if(block_row_indices_lens.size() != 2)
            MIGRAPHX_THROW("SparseAttention: block_row_indices input rank must be 2, actual: " +
                           std::to_string(block_row_indices_lens.size()));
        if((params.num_heads % params.num_layouts) != 0)
            MIGRAPHX_THROW("SparseAttention: block_row_indices input dim 0(num_layout) must be a "
                           "factor of num_heads, num_layout: " +
                           std::to_string(params.num_layouts) +
                           ", num_heads: " + std::to_string(params.num_heads));
        if(block_row_indices_lens[1] == 1)
            MIGRAPHX_THROW("SparseAttention: block_row_indices input dim 1 must be greater than 1");

        auto block_col_indices_lens = block_col_indices->get_shape().lens();
        if(block_col_indices_lens.size() != 2)
            MIGRAPHX_THROW("SparseAttention: block_col_indices input rank must be 2, actual: " +
                           std::to_string(block_col_indices_lens.size()));
        if(block_col_indices_lens[0] != params.num_layouts)
            MIGRAPHX_THROW("SparseAttention: block_col_indices input dim 0 must be equal to "
                           "block_row_indices dim 0, actual and expected values: " +
                           std::to_string(block_col_indices_lens[0]) + ", " +
                           std::to_string(params.num_layouts));
        if(block_col_indices_lens[1] > params.max_blocks * params.max_blocks)
            MIGRAPHX_THROW("SparseAttention: block_col_indices input dim 1 must be less than or "
                           "equal to max_block*max_blocks, actual: " +
                           std::to_string(block_col_indices_lens[1]));

        if((total_sequence_length->get_shape().ndim() != 1 or
            total_sequence_length->get_shape().elements() != 1) and
           not total_sequence_length->get_shape().scalar())
            MIGRAPHX_THROW(
                "SparseAttention: total_sequence_length input must be a scalar, actual dims: " +
                to_string_range(total_sequence_length->get_shape().lens()));

        if(key_total_sequence_lengths->get_shape().ndim() != 1 or
           key_total_sequence_lengths->get_shape().lens()[0] != params.batch_size)
            MIGRAPHX_THROW("SparseAttention: key_total_sequence_lengths input must be a vector of "
                           "length equal to batch_size: " +
                           std::to_string(params.batch_size) + ", actual: " +
                           std::to_string(key_total_sequence_lengths->get_shape().lens()[0]));
    }
    instruction_ref unpack_block_masks(module& mod,
                                       instruction_ref ins,
                                       instruction_ref block_row_ind,
                                       instruction_ref block_col_ind) const
    {
        const uint32_t num_layouts = block_row_ind->get_shape().lens()[0];
        const uint32_t mat_dim     = block_row_ind->get_shape().lens()[1] - 1;
        const uint32_t max_nnz     = block_col_ind->get_shape().lens()[1];
        const auto dtype           = block_row_ind->get_shape().type();
        const auto out_type        = shape::uint8_type;

        block_col_ind =
            mod.insert_instruction(ins, make_op("unsqueeze", {{"axes", {1}}}), block_col_ind);
        auto col_idx_lens = block_col_ind->get_shape().lens();
        col_idx_lens[1]   = mat_dim;
        block_col_ind     = mod.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", col_idx_lens}}), block_col_ind);

        const auto make_bounds = [&](const auto start, const auto end) {
            auto bounds = mod.insert_instruction(
                ins,
                make_op("slice", {{"axes", {1}}, {"starts", {start}}, {"ends", {end}}}),
                block_row_ind);
            bounds = mod.insert_instruction(ins, make_op("unsqueeze", {{"axes", {2}}}), bounds);
            auto bounds_lens = bounds->get_shape().lens();
            bounds_lens[2]   = max_nnz;
            return mod.insert_instruction(
                ins, make_op("multibroadcast", {{"out_lens", bounds_lens}}), bounds);
        };
        auto lower_bounds = make_bounds(0, mat_dim);
        auto upper_bounds = make_bounds(1, mat_dim + 1);

        std::vector<uint32_t> column_indices_vals(max_nnz);
        std::iota(column_indices_vals.begin(), column_indices_vals.end(), 0);
        auto column_indices =
            mod.insert_literal(ins, {shape{dtype, {1, 1, max_nnz}}, column_indices_vals});
        column_indices = mod.insert_instruction(
            ins,
            make_op("multibroadcast", {{"out_lens", lower_bounds->get_shape().lens()}}),
            column_indices);

        auto gte = mod.insert_instruction(ins, make_op("less"), column_indices, lower_bounds);
        gte      = mod.insert_instruction(ins, make_op("not"), gte);
        gte      = mod.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::bool_type}}), gte);
        auto lt = mod.insert_instruction(ins, make_op("less"), column_indices, upper_bounds);
        lt      = mod.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::bool_type}}), lt);
        auto where_predicate = mod.insert_instruction(ins, make_op("logical_and"), gte, lt);
        auto updates         = mod.insert_instruction(
            ins, make_op("convert", {{"target_type", out_type}}), where_predicate);

        auto out_of_bound_idx =
            mod.insert_literal(ins, {shape{dtype, {1, 1, 1}}, std::vector<uint32_t>{mat_dim}});
        out_of_bound_idx = mod.insert_instruction(
            ins,
            make_op("multibroadcast", {{"out_lens", block_col_ind->get_shape().lens()}}),
            out_of_bound_idx);

        auto indices = mod.insert_instruction(
            ins, make_op("where"), where_predicate, block_col_ind, out_of_bound_idx);

        auto out_mat =
            mod.insert_literal(ins, {shape{out_type, {1, 1, 1}}, std::vector<uint32_t>(0)});
        out_mat = mod.insert_instruction(
            ins,
            make_op("multibroadcast", {{"out_lens", {num_layouts, mat_dim, mat_dim}}}),
            out_mat);

        return mod.insert_instruction(
            ins,
            make_op("scatter_none", {{"axis", 2}, {"skip_out_of_bounds", true}}),
            out_mat,
            indices,
            updates);
    }

    instruction_ref make_block_masks(module& mod,
                                     instruction_ref ins,
                                     instruction_ref block_row_ind,
                                     instruction_ref block_col_ind,
                                     size_t sparse_block_size,
                                     const std::vector<size_t>& bnsm) const
    {
        auto masks = unpack_block_masks(mod, ins, block_row_ind, block_col_ind);
        // Want masks to go from:
        // {num_layouts, max_blocks, max_blocks}
        // to:
        // {batch_size, num_layouts * head_layout_factor, max_blocks * block_size, max_blocks *
        // block_size}
        // Where head_layout_factor is (num_heads + num_layouts - 1) / num_layouts
        // In dimension 1(num_layouts * head_layout_factor) the layouts need to be repeated, that
        // is: {layout_1, layout_2, ..., layout_n} -> {layout_1, layout_2, ..., layout_n, layout_1,
        // layout_2, ..., layout_n, ...}
        auto num_layouts        = masks->get_shape().lens()[0];
        auto head_layout_factor = (bnsm[1] + num_layouts - 1) / num_layouts;
        auto expanded_lens      = masks->get_shape().lens();
        expanded_lens.insert(expanded_lens.begin(), bnsm[0]);
        expanded_lens[1] *= head_layout_factor;
        expanded_lens[2] *= sparse_block_size;
        expanded_lens[3] *= sparse_block_size;
        auto expanded_masks =
            mod.insert_instruction(ins, make_op("unsqueeze", {{"axes", {0, 3, 5}}}), masks);

        auto bc_lens = expanded_masks->get_shape().lens();
        bc_lens[0]   = head_layout_factor;
        bc_lens[3]   = sparse_block_size;
        bc_lens[5]   = sparse_block_size;
        bc_lens.insert(bc_lens.begin(), bnsm[0]);
        expanded_masks = mod.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", bc_lens}}), expanded_masks);

        expanded_masks = mod.insert_instruction(
            ins, make_op("reshape", {{"dims", expanded_lens}}), expanded_masks);

        std::vector<int64_t> axes;
        std::vector<int64_t> starts;
        std::vector<int64_t> ends;
        if(expanded_masks->get_shape().lens()[2] > bnsm[2])
        {
            axes.push_back(2);
            // TODO - -2 ... -1 is incorrect. The range  to slice from should actually be
            // [past_sequence_length, past_sequence_length + 1) The problem is that
            // past_sequence_length can only be determined at runtime from key_total_seq_lens.
            starts.push_back(bnsm[2] == 1 ? -2 : 0);
            ends.push_back(bnsm[2] == 1 ? -1 : bnsm[2]);
        }
        if(expanded_masks->get_shape().lens()[3] > bnsm[3])
        {
            axes.push_back(3);
            starts.push_back(0);
            ends.push_back(bnsm[3]);
        }
        if(not axes.empty())
        {
            expanded_masks = mod.insert_instruction(
                expanded_masks,
                make_op("slice", {{"axes", axes}, {"starts", starts}, {"ends", ends}}),
                expanded_masks);
        }

        return mod.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::bool_type}}), expanded_masks);
    }

    instruction_ref make_causal_mask(module& mod,
                                     instruction_ref ins,
                                     const std::vector<size_t>& bnsm,
                                     instruction_ref ktsl) const
    {
        auto dtype = ktsl->get_shape().type();

        auto sl = mod.insert_literal(ins, {{ktsl->get_shape().type(), {1}}, {bnsm[2]}});
        sl      = mod.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", ktsl->get_shape().lens()}}), sl);

        auto psl = mod.insert_instruction(ins, make_op("sub"), ktsl, sl);
        psl = mod.insert_instruction(ins, make_op("reshape", {{"dims", {bnsm[0], 1, 1, 1}}}), psl);
        psl = mod.insert_instruction(ins, make_op("multibroadcast", {{"out_lens", bnsm}}), psl);

        std::vector<size_t> causal_lens_vals(bnsm[2]);
        std::iota(causal_lens_vals.begin(), causal_lens_vals.end(), 0);
        // Have to do literal->reshape->broadcast instead of literal with appropriate shape ->
        // broadcast to avoid simplify algebra messing up
        auto causal_lens = mod.insert_literal(ins, {{dtype, {bnsm[2]}}, causal_lens_vals});
        causal_lens =
            mod.insert_instruction(ins, make_op("reshape", {{"dims", {bnsm[2], 1}}}), causal_lens);
        causal_lens = mod.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", bnsm}}), causal_lens);

        causal_lens = mod.insert_instruction(ins, make_op("add"), causal_lens, psl);

        std::vector<size_t> column_indices_vals(bnsm[3]);
        std::iota(column_indices_vals.begin(), column_indices_vals.end(), 0);
        auto column_indices =
            mod.insert_literal(ins, {{dtype, {1, 1, 1, bnsm[3]}}, column_indices_vals});
        column_indices = mod.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", bnsm}}), column_indices);

        auto causal_mask =
            mod.insert_instruction(ins, make_op("greater"), column_indices, causal_lens);
        causal_mask = mod.insert_instruction(
            ins, make_op("convert", {{"target_type", shape::bool_type}}), causal_mask);
        return causal_mask = mod.insert_instruction(ins, make_op("not"), causal_mask);
    }

    instruction_ref attention_probabilities(
        module& mod, instruction_ref ins, instruction_ref q, instruction_ref k, float scale) const
    {
        k = mod.insert_instruction(ins, make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);
        auto attn_probs = mod.insert_instruction(ins, make_op("dot"), q, k);
        scale           = float_equal(scale, 0.0f)
                              ? 1.0f / std::sqrt(static_cast<float>(q->get_shape().lens()[3]))
                              : scale;
        auto scale_lit  = mod.insert_literal(ins, literal{shape{shape::float_type, {1}}, {scale}});
        scale_lit       = mod.insert_instruction(
            ins,
            make_op("multibroadcast", {{"out_lens", attn_probs->get_shape().lens()}}),
            scale_lit);
        return mod.insert_instruction(ins, make_op("mul"), attn_probs, scale_lit);
    }

    instruction_ref masked_softmax(module& mod,
                                   instruction_ref ins,
                                   instruction_ref attn_probs,
                                   instruction_ref block_row_indices,
                                   instruction_ref block_col_indices,
                                   size_t sparse_block_size,
                                   instruction_ref key_total_seq_lens,
                                   const std::vector<size_t>& bnsm) const
    {
        auto ninf = mod.insert_literal(
            ins,
            {{attn_probs->get_shape().type(), {1}}, {-std::numeric_limits<float>::infinity()}});
        ninf = mod.insert_instruction(
            ins, make_op("multibroadcast", {{"out_lens", attn_probs->get_shape().lens()}}), ninf);

        auto block_mask = make_block_masks(
            mod, ins, block_row_indices, block_col_indices, sparse_block_size, bnsm);
        auto causal_mask = make_causal_mask(mod, ins, bnsm, key_total_seq_lens);
        // TODO: Add mask to mask out elements based on key_total_seq_lens
        auto final_mask =
            mod.insert_instruction(ins, make_op("logical_and"), block_mask, causal_mask);
        attn_probs = mod.insert_instruction(ins, make_op("where"), final_mask, attn_probs, ninf);

        return mod.insert_instruction(ins, make_op("softmax", {{"axis", 3}}), attn_probs);
    }

    instruction_ref attention_scores(module& mod,
                                     instruction_ref ins,
                                     instruction_ref softmax,
                                     instruction_ref v) const
    {
        auto attn_scores = mod.insert_instruction(ins, make_op("dot"), softmax, v);
        attn_scores      = mod.insert_instruction(
            ins, make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), attn_scores);
        return mod.insert_instruction(
            ins,
            make_op("reshape", {{"dims", ins->outputs()[0]->get_shape().lens()}}),
            attn_scores);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
