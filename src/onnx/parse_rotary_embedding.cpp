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

struct rotary_parameters
{
    // Extracted from Inputs
    std::size_t batch_size           = 0; // Batch used by input
    std::size_t seq_len              = 0; // Sequence length used by input
    std::size_t head_size            = 0; // Head size used for offset in each block
    std::size_t max_seq_len          = 0; // Sequence length used by sin/cos caches
    std::size_t hidden_size          = 0; // Hidden size used by the input

    // Extracted from both
    std::size_t num_heads            = 0; // num_heads = hidden_size / head_size  or input

    // Extracted from Attributes  
    std::size_t rotary_embedding_dim = 0;
    bool interleaved                 = false; 
    bool is_packed_batching          = false;
    float scale                      = 0.0;
}

struct parse_rotary_embedding : op_parser<parse_rotary_embedding>
{
    std::vector<op_desc> operators() const { return {{"RotaryEmbedding"}}; }

    static void parse_attributes(const onnx_parser& parser, 
                                 const onnx_parser::node_info& info,
                                 rotary_parameters& param)

    {
        if(contains(info.attributes, "interleaved"))
        {
            param.interleaved =
                parser.parse_value(info.attributes.at("interleaved")).at<bool>();
        }

        if(contains(info.attributes, "is_packed_batching"))
        {
            param.is_packed_batching =
                parser.parse_value(info.attributes.at("is_packed_batching")).at<bool>();
        }

        if(contains(info.attributes, "num_heads"))
        {
            param.num_heads = parser.parse_value(info.attributes.at("num_heads")).at<std::size_t>();
        }

        if(contains(info.attributes, "rotary_embedding_dim"))
        {
            param.rotary_embedding_dim = parser.parse_value(info.attributes.at("rotary_embedding_dim")).at<std::size_t>();
        }

        if(contains(info.attributes, "scale"))
        {
            param.scale = parser.parse_value(info.attributes.at("scale")).at<float>();
        }

        if((param.num_heads != 0 xor param.rotary_embedding_dim != 0))
        {
            MIGRPHAX_THROW("RotaryEmbedding: num_heads and rotary_embedding dims must be used together and non-zero");
        }
    }

    static void parse_input(const instruction_ref& input,
                                  rotary_parameters& param)
    {
        auto input_lens  = input->get_shape().lens();
        auto input_dims  = input_lens.size();

        if(input_dims < 3 or input_dims > 4)
        {
            MIGRAPHX_THROW("RotaryEmbedding:Input must be 3D (Batch , Sequence Length, Hidden size) or \
                            4D (Batch, Num Heads, Sequence Length, Head Size))");
        }

        param.batch = input_lens.at(0);

        if (input_dims == 3)
        {
            param.seq_len = input_lens.at(1);
            param.hidden_size = input_lens.at(2);
        }
        else
        {
            param.num_heads = input_lens.at(1);
            param.seq_len = input_lens.at(2);
            param.head_size = input_lens.at(3);
        }
    }

    // Ensure position ID shapes comply with input dimensions 
    static void parse_position_ids(const instruction_ref& position_ids,
                                   const rotary_parameters& param)
   {
        auto position_len = position_ids->get_shape().lens();
        auto position_dim = position_ids->get_shape().lens().size();

        if(position_dim > 2 or position_ids.get_shape().scalar())
        {
            MIGRAPHX_THROW("RotaryEmbedding: Position_ids must be either 1D tensor of shape (1) or 2d (Batch, Sequence Length)");
        }

        if(position_dim == 1 and position_len.at(0) != 1)
        {
            MIGRAPHX_THROW("RotaryEmbedding: Position_id must have shape of 1 for 1D tensor")
        }

        if(position_dim == 2 and position_len.at(0) != param.batch or position_len.at(1) != param.seq_len)
        {
            MIGRAPHX_THROW("RotaryEmbedding: Position_id 2D dims must match input batch size and sequenec length");
        }
    }

    static void parse_cos_cache(const instruction_ref& cos_cache,
                                      rotary_parameters& param)
    {
        auto cos_cache_len = cos_cache->get_shape().lens();
        param.max_seq_len = cos_cache_len.at(0);

        compsare_sin_cos_cache_dims(cos_cache_len.at(1));
    }

    static void parse_sin_cache(const instruction_ref& sin_cache,
                                const rotary_parameters& param)
    {
        auto sin_cache_len = sin_cache->get_shape().lens();

        if(param.max_seq_len != sin_cache_len.at(0))
        {
            MIGRAPHX_THROW("RotaryEmbedding: max_sequence_length must be the same between sin & cos caches!");
        }

        compsare_sin_cos_cache_dims(sin_cache_len.at(1));
    }

    static void compare_sin_cos_cache_dims(const size_t dim,
                                           const rotary_parameters &param)
    {
        if(param.rotary_embedding_dim != 0)
        {
            if(param.rotary_embedding_dim / 2 != dim)
            {
                MIGRAPHX_THROW("RotaryEmbedding: rotary_embedding must be the same between sin & cos caches!");
            }
        }
        else   
        {
            if(param.head_size / 2  != dim)
            {
                MIGRAPHX_THROW("RotaryEmbedding: head size for sin & cos caches must match input");
            }
        }
    }


    static void parse_input_args(const std::vector<instruction_ref>& args, 
                            rotary_parameters& param)
    {
        // Order matters as we're basing params related to the first input
        parse_input(args.at(0), param);
        parse_position_ids(args.at(1), param);       
        parse_cos_cache(args.at(2), param;
        parse_sin_cache(args.at(3), param;


    }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        rotary_parameters params{};

        if(args.size() != 4)
        {
            MIGRAPHX_THROW("RotaryEmbedding: Wrong number of inputs provided require 4");
        }

        // Sanity check input dimension and shapes while extracting params
        parse_attributes(parse, info, params);
        parse_input_args(args, params);

        // Setup based on parsed params gathered from input attributes/inputs
        auto input        = args.at(0)
        auto position_ids = args.at(1)
        auto cos_cache    = args.at(2);
        auto sin_cache    = args.at(3);


        return output;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
