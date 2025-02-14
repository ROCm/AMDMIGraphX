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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {



struct parse_attention : op_parser<parse_attention>
{
    std::vector<op_desc> operators() const { return {{"Attention"}}; }

    struct attention_attr
    {
        bool do_rotary           = false;
        bool past_present_share_buffer = false;
        bool unidirectional      = false; 
        std::size_t num_heads    = 1;         //required by inputs 
        std::size_t rotary_embedding_dim = 0; // Gets set to head_size when not set
        std::vector<std::size_t> qkv_num_heads{0, 0, 0}; //Sets hidden sizes if not set defiend by input
        float scale              = 0.0;       // Default should be 1/sqrt(head_size)
        float mask_filter_val    = -10000.0f;
    }

    static void handle_attributes(const onnx_parser::node_info& info
                                  struct attention_attr& attr_out)
    {
        if(contains(info.attributes, "do_rotary"))
        {
            attr_out.do_rotary = parser.parse_value(info.attributes.at("do_rotary")).at<bool>();
        }

        if(contains(info.attributes, "mask_filter_value"))
        {
            attr_out.mask_filter_val = parser.parse_value(info.attributes.at("mask_filter_value")).at<float>();
        }

        if(contains(info.attributes, "num_heads"))
        {
            attr_out.num_heads = parser.parse_value(info.attributes.at("num_heads")).at<std::size_t>();
        }
        else
        {
            MIGRAPHX_THROW("PARSE_ATTENTION: num_heads attribute required");
        }       

        if(contains(info.attributes, "past_present_share_buffer"))
        {
            attr_out.past_present_share_buffer = 
                (1 == parser.parse_value(info.attributes.at("past_present_share_buffer")).at<size_t>());
        }

        if(contains(info.attributes, "qkv_hidden_sizes"))
        {
           attr_out.qkv_num_heads = parser.parse_value(info.attributes.at("qkv_hidden_sizes")).at<std::vector<int>(3)>();
        }

        if(contains(info.attributes, "rotary_embedding_dim"))
        {
            auto rotary_embedding_dim =
                parser.parse_value(info.attributes.at("rotary_embedding_dim")).at<size_t>();
            if(rotary_embedding_dim != 32 and rotary_embedding_dim != 64 and rotary_embedding_dim != 128)
            {
                MIGRAPHX_THROW("PARSE_ATTENTION: rotary_embedding_dim must be either 32, 64, or 128");
            }
            attr_out.rotary_embedding_dim = rotary_embedding_dim;
        }
        if(contains(info.attributes, "scale"))
        {
            attr_out.scale = parser.parse_value(info.attributes.at("scale")).at<float>();
        }

        if(contains(info.attributes, "unidirectional"))
        {
            attr_out.unidirectional = (1 == parser.parse_value(info.attributes.at("unidirectional")).at<size_t>());
        }
    }


    static void handle_arguments(const std::vector<instruction_ref>& args,
                                 struct attention_attr& attr_out)
    {
        if(args.size() < 2 or args.size() > 7)
        {
            MIGRAPHX_THROW("Attention: Wrong number of inputs provided");
        }
    }

    static instruction_ref scale_dot_attention_head(const onnx_parser::node_info& info,
                                                    const instruction_ref& Q,
                                                    const instruction_ref& K,
                                                    const instruction_ref& V)
    {


        return scale_output;
    }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        struct attention_attr parsed_attributes;
        handle_attributes(info, parsed_attributes);
        handle_arguments(args, parsed_attributes);

        return {};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
