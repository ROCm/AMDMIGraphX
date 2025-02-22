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
        bool do_rotary                 = false;
        bool past_present_share_buffer = false;
        bool unidirectional            = false; 
        std::size_t num_heads          = 1;   //required by inputs 
        std::size_t rotary_embedding_dim = 0; // Gets set to head_size when not set
        std::vector<std::size_t> qkv_hidden_sizes{0, 0, 0}; //Sets hidden sizes if not set defiend by input
        float scale              = 0.0;       // Default should be 1/sqrt(head_size)
        float mask_filter_val    = -10000.0f;
    };

    // Values infered from input vectors
    struct attention_infered
    {
        std::size_t batch_size;        // Pull from input
        std::size_t input_hidden_size; // Pulled from input and/or weights
        std::size_t hidden_size;   // Pulled from weights vector (weights.at(1) / 2)
        std::size_t v_hidden_size; // Value weight size
        std::size_t sequence_length;
        float head_size; // Used for the scale factor of attention stages by default if scale is not defined 
    };

    static void handle_attributes(const onnx_parser& parser,
                                  const onnx_parser::node_info& info,
                                  struct attention_attr& attr_out,
                                  struct attention_infered& infered_out)
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
            auto input_val = parser.parse_value(info.attributes.at("qkv_hidden_sizes"));

            /* auto q_size = input_val.data().at(0);
            auto k_size = input_val.data().at(1);
            auto v_size = input_val.data().at(2);

            if(q_size != k_size)
            {
                MIGRAPHX_THROW("Attention: q and k hidden sizes must be identitcal!");
            } */

            //attr_out.qkv_hidden_sizes = input_val->.lens();
            /*if(std::any_of(qkv_vec.begin(), qkv_vec.end(), [](auto i){return (i == 0) or (i < 0);}))
            {
                MIGRAPHX_THROW("PARSE_ATTENTION: qkv_hidden_sizes must be nonzero and valid");
            }
            attr_out.qkv_hidden_sizes = qkv_vec;*/
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

    static void qkv_sizes_sum_arg_valid(const std::vector<size_t>& qkv_vec,
                                        const instruction_ref input_arg,
                                        const size_t dim,
                                        const std::string name)
    {
        if(std::accumulate(qkv_vec.begin(), qkv_vec.end(), 0) != input_arg->get_shape().lens().at(dim))
        {
            MIGRAPHX_THROW("Attention: q k v hidden sizes sum must match" + name + " tensor" + std::to_string(dim) + "dimension");
        }
    }

    static void handle_projection_bias(const std::vector<instruction_ref>& args,
                                       struct attention_attr& attr_out,
                                       struct attention_infered& infered_out,
                                       std::vector<instruction_ref>& output_arg_vec)
    {
    }

    static void handle_mask_index(const std::vector<instruction_ref>& args,
                                  struct attention_attr& attr_out,
                                  struct attention_infered& infered_out,
                                  std::vector<instruction_ref>& output_arg_vec)
    {

    }

    static void handle_past(const std::vector<instruction_ref>& args,
                            struct attention_attr& attr_out,
                            struct attention_infered& infered_out,
                            std::vector<instruction_ref>& output_arg_vec)
    {

    }
    
    static void handle_attention_bias(const std::vector<instruction_ref>& args,
                                      struct attention_attr& attr_out,
                                      struct attention_infered& infered_out,
                                      std::vector<instruction_ref>& output_arg_vec)
    {

    }

    static void handle_past_sequence_length(const std::vector<instruction_ref>& args,
                                            struct attention_attr& attr_out,
                                            struct attention_infered& infered_out,
                                            std::vector<instruction_ref>& output_arg_vec)
    {

    }

    static std::vector<instruction_ref> handle_arguments(const onnx_parser& parser,
                                                         const std::vector<instruction_ref>& args,
                                                         struct attention_attr& attr_out,
                                                         struct attention_infered& infered_out)
    {
        std::vector<instruction_ref> input_arguments;

        if(args.size() < 2 or args.size() > 7)
        {
            MIGRAPHX_THROW("Attention: Wrong number of inputs provided");
        }

        auto input_tensor  = args.at(0);
        auto input_shape   = input_tensor->get_shape();
        auto weight_tensor = args.at(1);
        auto weight_shape  = weight_tensor->get_shape();

        if(weight_shape.lens().at(0) != input_shape.lens().at(2))
        {
            MIGRAPHX_THROW("Attention: Input hidden size must be the same for input and weight tensors");
        }

        infered_out.batch_size        = input_shape.lens().at(0);
        infered_out.sequence_length   = input_shape.lens().at(1);
        infered_out.input_hidden_size = input_shape.lens().at(2);
        input_arguments.push_back(input_tensor);

        // Ensure qkv_hidden sizes set are valid wrt to input weights
        qkv_sizes_sum_arg_valid(attr_out.qkv_hidden_sizes, weight_tensor, 2, "weights");
        input_arguments.push_back(weight_tensor);

        // Handle theses individually. Order matters here to check conditions
        handle_projection_bias(args, attr_out, infered_out, input_arguments);
        handle_mask_index(args, attr_out, infered_out, input_arguments);
        handle_past(args, attr_out, infered_out, input_arguments);
        handle_attention_bias(args, attr_out, infered_out, input_arguments);
        handle_past_sequence_length(args, attr_out, infered_out, input_arguments);

        return input_arguments;
    }

    static instruction_ref scale_dot_attention_head(const onnx_parser::node_info& info,
                                                    const instruction_ref& Q,
                                                    const instruction_ref& K,
                                                    const instruction_ref& V,
                                                    const instruction_ref& scale_factor,
                                                    const instruction_ref& mask,
                                                    bool masked=false)
    {
        auto qk_out = info.add_instruction(make_op("dot"), Q, K);
        auto qk_scaled = info.add_instruction(make_op("div"), qk_out, scale_factor);
        auto qk_masked = qk_scaled;

        if(masked)
            qk_masked = info.add_instruction(make_op("dot"), qk_scaled, mask);

        auto softmax_out = info.add_instruction(make_op("softmax"), qk_masked);
        return info.add_instruction(make_op("dot"), softmax_out, V);
    }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        struct attention_attr parsed_attributes;
        struct attention_infered infered_attributes;

        handle_attributes(parser, info, parsed_attributes, infered_attributes);
        auto inputs = handle_arguments(parser, args, parsed_attributes, infered_attributes);

        instruction_ref q;
        instruction_ref k;
        instruction_ref v;
        instruction_ref mask;
        bool has_mask = false;
        instruction_ref present;

        // Used to scale all key values before any masking or other inputs
        auto scale_factor = info.add_literal(migraphx::literal{migraphx::shape{k->get_shape().type()}, {std::sqrt(k->get_shape().elements()) } } );

        //Get vector of attention heads and then concat the output results
        std::vector<instruction_ref> vec_of_attn_outs(parsed_attributes.num_heads);
        std::transform(vec_of_attn_outs.begin(),
                       vec_of_attn_outs.end(),
                       vec_of_attn_outs.begin(),
                       std::back_inserter(vec_of_attn_outs),
                       [&](auto&&, auto&& ) {
                           return scale_dot_attention_head(info, q, k, v, scale_factor, mask, has_mask);
                        });
        auto output = info.add_instruction(make_op("concat"), vec_of_attn_outs);

        std::vector<instruction_ref> output_vec{};
        output_vec.push_back(output);

        if(parsed_attributes.past_present_share_buffer)
            output_vec.push_back(present);

        return output_vec;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
