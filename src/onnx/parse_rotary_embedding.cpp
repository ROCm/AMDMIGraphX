/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_rotary_embedding : op_parser<parse_rotary_embedding>
{
    std::vector<op_desc> operators() const { return {{"RotaryEmbedding"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        bool interleaved                 = false;
        bool is_packed_batching          = false;
        std::size_t num_heads            = 0;
        std::size_t rotary_embedding_dim = 0;
        float scale                      = 0.0;

        if(contains(info.attributes, "interleaved"))
        {
            interleaved =
                parser.parse_value(info.attributes.at("interleaved")).at<bool>();
        }

        if(contains(info.attributes, "is_packed_batching"))
        {
            local_window_size =
                parser.parse_value(info.attributes.at("is_packed_batching")).at<bool>();
        }
        if(contains(info.attributes, "num_heads"))
        {
            num_heads = parser.parse_value(info.attributes.at("num_heads")).at<std::size_t>();
        }

        if(contains(info.attributes, "rotary_embedding_dim"))
        {
            rotary_embedding_dim = parser.parse_value(info.attributes.at("rotary_embedding_dim")).at<std::size_t>();
        }

        if(contains(info.attributes, "scale"))
        {
            scale = parser.parse_value(info.attributes.at("scale")).at<float>();
        }

        if(args.size() < 4)
        {
            MIGRAPHX_THROW("RotaryEmbedding: Wrong number of inputs provided require 4");
        }

        auto output             = info.add_instruction(make_op("RotaryEmbedding",
                                                            {{"interleved", interleaved},
                                                             {"is_batched_packing", is_batched_packing},
                                                             {"num_heads", num_heads},
                                                             {"rotary_embedding_dim", rotary_embedding_dim},
                                                             {"scale", scale}}), args);
        return output;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
