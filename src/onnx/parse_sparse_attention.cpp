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
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_sparse_attention : op_parser<parse_sparse_attention>
{

    std::vector<op_desc> operators() const { return {{"SparseAttention"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        if(not contains(info.attributes, "num_heads"))
            MIGRAPHX_THROW("SparseAttention: num_heads attribute is required");
        if(not contains(info.attributes, "kv_num_heads"))
            MIGRAPHX_THROW("SparseAttention: kv_num_heads attribute is required");
        if(not contains(info.attributes, "sparse_block_size"))
            MIGRAPHX_THROW("SparseAttention: sparse_block_size attribute is required");

        int64_t num_heads    = parser.parse_value(info.attributes.at("num_heads")).at<int>();
        int64_t kv_num_heads = parser.parse_value(info.attributes.at("kv_num_heads")).at<int>();
        int64_t sparse_block_size =
            parser.parse_value(info.attributes.at("sparse_block_size")).at<int>();

        if(args.size() < 9 or args.size() > 11)
            MIGRAPHX_THROW("SparseAttention: Wrong number of inputs");

        auto query = args[0];
        auto query_lens = query->get_shape().lens();
        instruction_ref key;
        instruction_ref value;

        int64_t head_size;

        if(args[1]->is_undefined())
        {
            if(!args[2]->is_undefined())
                MIGRAPHX_THROW("SparseAttention: Input 'key' and 'value' shall be both present or both absent");

            // qkv packed: (batch_size, sequence_length, d), where d is (num_heads + 2 * kv_num_heads) * head_size

            head_size = query_lens[2] / (num_heads + 2 * kv_num_heads);
        }
        else
        {
            if(args[2]->is_undefined())
                MIGRAPHX_THROW("SparseAttention: Input 'key' and 'value' shall be both present or both absent");

            key = args[1];
            value = args[2];
            head_size = query_lens[2] / num_heads;
        }

        float scale = 1 / std::sqrt(head_size);
        if(contains(info.attributes, "scale"))
            scale = parser.parse_value(info.attributes.at("scale")).at<float>();
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
