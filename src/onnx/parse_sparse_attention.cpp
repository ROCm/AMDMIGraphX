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
    std::vector<op_desc> operators() const { return {{"SparseAttention"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        auto num_heads = parse_attribute<int>(info, parser, "num_heads", true);
        if(num_heads <= 0)
            MIGRAPHX_THROW("SparseAttention: num_heads must be greater than zero, actual: " +
                           std::to_string(num_heads));
        auto kv_num_heads = parse_attribute<int>(info, parser, "kv_num_heads", true);
        if(kv_num_heads <= 0)
            MIGRAPHX_THROW("SparseAttention: kv_num_heads must be greater than zero, actual: " +
                           std::to_string(kv_num_heads));
        auto sparse_block_size = parse_attribute<int>(info, parser, "sparse_block_size", true);
        if(sparse_block_size <= 0)
            MIGRAPHX_THROW(
                "SparseAttention: sparse_block_size must be greater than zero, actual: " +
                std::to_string(sparse_block_size));
        auto scale              = parse_attribute(info, parser, "scale", false, 0.0f);
        auto do_rotary          = parse_attribute(info, parser, "do_rotary", false, false);
        auto rotary_interleaved = parse_attribute(info, parser, "rotary_interleaved", false, false);

        auto num_args          = args.size();
        auto expected_num_args = do_rotary ? 11 : 9;
        if(num_args != expected_num_args)
            MIGRAPHX_THROW(
                "SparseAttention: when do_rotary=" + std::string{do_rotary ? "true" : "false"} +
                ", number of inputs should be " + std::to_string(expected_num_args) +
                ", actual: " + std::to_string(num_args));

        const auto qkv                        = args[0];
        const auto k                          = args[1];
        const auto v                          = args[2];
        const auto past_k                     = args[3];
        const auto past_v                     = args[4];
        const auto block_row_indices          = args[5];
        const auto block_col_indices          = args[6];
        const auto total_sequence_length      = args[7];
        const auto key_total_sequence_lengths = args[8];

        if(not(k->is_undefined() and v->is_undefined()))
            MIGRAPHX_THROW("SparseAttention: only packed qkv supported, key and value inputs must "
                           "be undefined");
        if(qkv->get_shape().ndim() != 3)
            MIGRAPHX_THROW("SparseAttention: query input rank must be 3, actual: " +
                           std::to_string(qkv->get_shape().ndim()));

        auto&& qkv_lens        = qkv->get_shape().lens();
        auto batch_size        = qkv_lens[0];
        auto sequence_length   = qkv_lens[1];
        auto hidden_size       = qkv_lens[2];
        size_t total_num_heads = num_heads + 2 * kv_num_heads;
        auto head_size         = hidden_size / total_num_heads;

        if((hidden_size % total_num_heads) != 0)
            MIGRAPHX_THROW("SparseAttention: QKV hidden size must be divisible by (num_heads + 2 * "
                           "kv_num_heads), actual hidden size: " +
                           std::to_string(hidden_size) + ", actual num_heads and kv_num_heads: " +
                           std::to_string(num_heads) + ", " + std::to_string(kv_num_heads));
        auto head_size_factor = do_rotary ? 16 : 8;
        if((head_size % head_size_factor) != 0)
            MIGRAPHX_THROW(
                "SparseAttention: when do_rotary=" + std::string{do_rotary ? "true" : "false"} +
                ", head_size must be a multiple of " + std::to_string(head_size_factor) +
                ", actual: " + std::to_string(head_size));

        auto&& past_k_lens = past_k->get_shape().lens();
        if(past_k_lens.size() != 4)
            MIGRAPHX_THROW("SparseAttention: past_key rank should be 4, actual: " +
                           std::to_string(past_k_lens.size()));
        if(past_k_lens[0] != batch_size)
            MIGRAPHX_THROW("SparseAttention: past_key input dim 0 must be equal to batch_size: " +
                           std::to_string(batch_size) +
                           ", actual: " + std::to_string(past_k_lens[0]));
        if(past_k_lens[1] != kv_num_heads)
            MIGRAPHX_THROW("SparseAttention: past_key input dim 1 must be equal to kv_num_heads: " +
                           std::to_string(kv_num_heads) +
                           ", actual: " + std::to_string(past_k_lens[1]));
        if(past_k_lens[3] != head_size)
            MIGRAPHX_THROW("SparseAttention: past_key input dim 3 must be equal to head_size: " +
                           std::to_string(head_size) +
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
        auto num_layout = block_row_indices_lens[0];
        if((num_heads % num_layout) != 0)
            MIGRAPHX_THROW("SparseAttention: block_row_indices input dim 0(num_layout) must be a "
                           "factor of num_heads, num_layout: " +
                           std::to_string(num_layout) +
                           ", num_heads: " + std::to_string(num_heads));
        if(block_row_indices_lens[1] == 1)
            MIGRAPHX_THROW("SparseAttention: block_row_indices input dim 1 must be greater than 1");

        auto block_col_indices_lens = block_col_indices->get_shape().lens();
        if(block_col_indices_lens.size() != 2)
            MIGRAPHX_THROW("SparseAttention: block_col_indices input rank must be 2, actual: " +
                           std::to_string(block_col_indices_lens.size()));
        if(block_col_indices_lens[0] != num_layout)
            MIGRAPHX_THROW("SparseAttention: block_col_indices input dim 0 must be equal to "
                           "block_row_indices dim 0, actual and expected values: " +
                           std::to_string(block_col_indices_lens[0]) + ", " +
                           std::to_string(num_layout));
        auto max_blocks = block_row_indices_lens[1] - 1;
        if(block_col_indices_lens[1] > max_blocks * max_blocks)
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
           key_total_sequence_lengths->get_shape().lens()[0] != batch_size)
            MIGRAPHX_THROW("SparseAttention: key_total_sequence_lengths input must be a vector of "
                           "length equal to batch_size: " +
                           std::to_string(batch_size) + ", actual: " +
                           std::to_string(key_total_sequence_lengths->get_shape().lens()[0]));

        if(do_rotary)
        {
            auto cos_cache = args[9];
            auto sin_cache = args[10];
            auto cos_dim1  = cos_cache->get_shape().lens()[1];
            if(cos_dim1 > head_size / 2 or cos_dim1 % 8 != 0)
                MIGRAPHX_THROW(
                    "SparseAttention: cos_cache input dim 1 must be a multiple of 8 and <= " +
                    std::to_string(head_size / 2) +
                    "(head_size/2), actual: " + std::to_string(cos_dim1));
            if(cos_cache->get_shape() != sin_cache->get_shape())
            {
                std::stringstream err_msg;
                err_msg << "SparseAttention: cos_cache and sin_cache inputs must have the same "
                           "shape, actual shapes for cos_cache and sin_cache: ";
                err_msg << cos_cache->get_shape() << ", " << sin_cache->get_shape();
                MIGRAPHX_THROW(err_msg.str());
            }
        }

        auto new_args = args;
        new_args[0]   = info.add_instruction(
            make_op(
                "reshape",
                {{"dims", {batch_size, sequence_length, num_heads + 2 * kv_num_heads, head_size}}}),
            args[0]);
        new_args[0] = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}),
                                           new_args[0]);
        auto sparse_attn = info.add_instruction(make_op("sparse_attention",
                                                        {{"do_rotary", do_rotary},
                                                         {"rotary_interleaved", rotary_interleaved},
                                                         {"num_heads", num_heads},
                                                         {"kv_num_heads", kv_num_heads},
                                                         {"scale", scale},
                                                         {"sparse_block_size", sparse_block_size}}),
                                                new_args);

        std::vector<instruction_ref> ret_ins;
        for(auto i = 0; i < sparse_attn->get_shape().sub_shapes().size(); ++i)
            ret_ins.push_back(
                info.add_instruction(make_op("get_tuple_elem", {{"index", i}}), sparse_attn));

        return ret_ins;
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
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
