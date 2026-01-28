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
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_attention : op_parser<parse_attention>
{
    std::vector<op_desc> operators() const { return {{"Attention"}}; }

    static void handle_qkv_hidden_size_attr(const onnx_parser& parser,
                                            const onnx_parser::node_info& info,
                                            value& options)
    {
        auto input_val = parser.parse_value(info.attributes.at("qkv_hidden_sizes"));
        std::vector<int64_t> qkv_values;

        if(input_val.get_shape().type() != shape::int64_type)
        {
            MIGRAPHX_THROW("PARSE_ATTENTION: qkv_hidden_sizes must be int64 type");
        }

        qkv_values = input_val.get_argument().to_vector<int64_t>();

        if(qkv_values.size() != 3)
        {
            MIGRAPHX_THROW("PARSE_ATTENTION: qkv_hidden_sizes must have exactly 3 values");
        }

        if(qkv_values[0] != qkv_values[1])
        {
            MIGRAPHX_THROW("PARSE_ATTENTION: q and k hidden sizes must be identical!");
        }

        std::vector<std::size_t> qkv_vec{static_cast<std::size_t>(qkv_values[0]),
                                         static_cast<std::size_t>(qkv_values[1]),
                                         static_cast<std::size_t>(qkv_values[2])};
        if(std::any_of(qkv_vec.begin(), qkv_vec.end(), [](auto i) { return i == 0; }))
        {
            MIGRAPHX_THROW("PARSE_ATTENTION: qkv_hidden_sizes must be nonzero and valid");
        }

        options.insert({"qkv_hidden_sizes", qkv_vec});
    }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        value options = {};

        if(contains(info.attributes, "do_rotary"))
        {
            bool do_rotary = (1 == parser.parse_value(info.attributes.at("do_rotary")).at<int>());
            options.insert({"do_rotary", do_rotary});
        }

        if(contains(info.attributes, "mask_filter_value"))
        {
            float mask_filter_val =
                parser.parse_value(info.attributes.at("mask_filter_value")).at<float>();
            options.insert({"mask_filter_val", mask_filter_val});
        }

        if(contains(info.attributes, "num_heads"))
        {
            std::size_t num_heads =
                parser.parse_value(info.attributes.at("num_heads")).at<std::size_t>();
            options.insert({"num_heads", num_heads});
        }
        else
        {
            MIGRAPHX_THROW("PARSE_ATTENTION: num_heads attribute required");
        }

        if(contains(info.attributes, "past_present_share_buffer"))
        {
            bool past_present_share_buffer =
                (1 == parser.parse_value(info.attributes.at("past_present_share_buffer"))
                          .at<std::size_t>());
            options.insert({"past_present_share_buffer", past_present_share_buffer});
        }

        if(contains(info.attributes, "qkv_hidden_sizes"))
        {
            handle_qkv_hidden_size_attr(parser, info, options);
        }

        if(contains(info.attributes, "rotary_embedding_dim"))
        {
            std::size_t rotary_embedding_dim =
                parser.parse_value(info.attributes.at("rotary_embedding_dim")).at<std::size_t>();

            if(rotary_embedding_dim != 32 and rotary_embedding_dim != 64 and
               rotary_embedding_dim != 128)
            {
                MIGRAPHX_THROW(
                    "PARSE_ATTENTION: rotary_embedding_dim must be either 32, 64, or 128");
            }

            options.insert({"rotary_embedding_dim", rotary_embedding_dim});
        }

        if(contains(info.attributes, "scale"))
        {
            float scale_val = parser.parse_value(info.attributes.at("scale")).at<float>();
            options.insert({"scale", scale_val});
        }

        if(contains(info.attributes, "unidirectional"))
        {
            bool unidirectional =
                (1 == parser.parse_value(info.attributes.at("unidirectional")).at<int>());
            options.insert({"unidirectional", unidirectional});
        }

        return op::builder::add("attention", *info.mod, args, options).at(0);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
