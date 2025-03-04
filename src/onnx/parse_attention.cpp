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

    enum mask_index_pad{ // Used to check mask_index when input vector size == 2
        NONE,            // Not Set - If input isn't set indicates this is an error with op
        RAW,             // Indicates input mask is raw mask where 0 masks the value and 1 does not.
        RIGHT_PADDING,   // second dimension is (batch_size)
        LEFT_PADDING,    // second dimension
    };

    enum attention_mode{  // Important to determine how to calculate total_sequence_length
        NOT_SET,          // Not Set - If Past/Present used this indicates an error with op
        SELF,             // Implies K,V lengths equaual and equal to sequence_length 
        CROSS_ATTENTION,  // Relevant as K/V may have different lengths in this case
    };

    // For items explicitly parsed from attributes
    struct attention_attr
    {
        bool do_rotary                 = false; // Rotary encode input prior to projection with weight matrices
        bool past_present_share_buffer = false; // Related to past input shares buffer with present output
        bool unidirectional            = false; // Mask is lower triangular ie) only pay attention to prev words
        std::size_t num_heads          = 1;     // Required by inputs 
        std::size_t rotary_embedding_dim = 0;   // Gets set to head_size when not set
        std::vector<std::size_t> qkv_hidden_sizes{0, 0, 0}; //Sets hidden sizes if not set defiend by input
        float scale              = 0.0;         // Default 1/sqrt(query_size) (query_size also known as head_size)
        float mask_filter_val    = -10000.0f;   // Default value used for masking our input encoding
    };

    // Values infered from input vectors or attributes
    struct attention_infered
    {
        std::size_t batch_size;            // From input dim(1)
        std::size_t input_hidden_size;     // From input dim(3) - Also known as embeddng_size in literature
        std::size_t hidden_size;           // From weights or qkv_hidden_sizes, related to q,k size which must be equal
        std::size_t v_hidden_size;         // From weights or qkv_hiddn_sizes - differs during cross attention_mode
        std::size_t sequence_length;       // Pulled from input dim(1), must be consistent between other parms
        std::size_t max_sequence_length;   // Pulled from mask_index 
        std::size_t total_sequence_length; // Pulled from past_seq_length + sequence_length
        std::size_t query_size;            // Also known as head_size. Derived via num_heads
        bool has_past_input = false;       // Set to true when we have past input. Present output required when set
        bool has_input_bias = false;       // Set to true when we have input_bias
        bool has_attn_mask = false;        // Set to true when we have an attention mask set
        bool has_attn_bias = false;        // Set to true when we have an attention bias input
        bool scale_not_query_sz = false;   // Set when a different scale attribute from 1/sqrt(query_size)
        enum mask_index_pad index_pad;     // Used to track state of input projection mask padding
        enum attention_mode attn_type;     // Used to determine the attention configuration
    };

    static void handle_attributes(const onnx_parser& parser,
                                  const onnx_parser::node_info& info,
                                  struct attention_attr& attr_out,
                                  struct attention_infered& infered_out)
    {
        if(contains(info.attributes, "do_rotary"))
        {
            attr_out.do_rotary = (1 == parser.parse_value(info.attributes.at("do_rotary")).at<int>());
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
            auto input_val = parser.parse_value(info.attributes.at("qkv_hidden_sizes")).get_argument();
            /*auto q_size = std::const_cast<size_t>(input_val.element(0).data());
            auto k_size = std::const_cast<size_t>(input_val.element(1).data());
            auto v_size = std::const_cast<size_t>(input_val.element(2).data());
            std::vector<size_t> qkv_vec{q_size, k_size, v_size};

            if(q_size != k_size)
            {
                MIGRAPHX_THROW("Attention: q and k hidden sizes must be identitcal!");
            } 

            if(std::any_of(qkv_vec.begin(), qkv_vec.end(), [](auto i){return (i == 0) or (i < 0);}))
            {
                MIGRAPHX_THROW("PARSE_ATTENTION: qkv_hidden_sizes must be nonzero and valid");
            }

            attr_out.qkv_hidden_sizes = */
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
            infered_out.scale_not_query_sz = true;
        }

        if(contains(info.attributes, "unidirectional"))
        {
            attr_out.unidirectional = (1 == parser.parse_value(info.attributes.at("unidirectional")).at<size_t>());
        }
    }

    // qkv values must be greater than zero to be "set"
    static bool qkv_size_not_set(std::vector<size_t> &qkv_vec)
    {
        return std::any_of(qkv_vec.begin(), qkv_vec.end(), [](auto i){return i <= 0;});
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

    static bool weights_not_equal(shape& weight_shape)
    {
        return (weight_shape.lens().at(1) % 3 != 0);
    }

    // simple call to check if the arg index exists
    static bool check_and_return_arg(const std::vector<instruction_ref>& args,
                                     const size_t index,
                                     instruction_ref& output_arg)
    {
        if(args.size() > index)
        {
            output_arg = args.at(index);
            return true;
        }
        return false;
    }

    static void handle_input(const instruction_ref& input_arg,
                             const struct attention_attr& parsed_in,
                             struct attention_infered& infered_out,
                             std::vector<instruction_ref>& output_arg_vec)
    {
        auto input_tensor  = input_arg;
        auto input_shape   = input_tensor->get_shape();

        infered_out.batch_size        = input_shape.lens().at(0);
        infered_out.sequence_length   = input_shape.lens().at(1);
        infered_out.input_hidden_size = input_shape.lens().at(2);
        // Determine the query_size used to generate attention heads that operate on each Q, K, V 
        // matrix.
        infered_out.query_size = infered_out.sequence_length / parsed_in.num_heads;
        infered_out.total_sequence_length = infered_out.sequence_length;

        output_arg_vec.push_back(input_tensor);

    }

    static void handle_weight(const instruction_ref& weight_arg,
                              const instruction_ref& input_arg,
                              struct attention_attr& attr_out,
                              struct attention_infered& infered_out,
                             std::vector<instruction_ref>& output_arg_vec)
    {
        auto weight_tensor = weight_arg;
        auto weight_shape  = weight_tensor->get_shape();
        auto input_shape   = input_arg->get_shape();

        if(weight_shape.lens().at(0) != input_shape.lens().at(2))
        {
            MIGRAPHX_THROW("Attention: Input hidden size must be the same for input and weight tensors");
        }

        if(weight_shape.type() != input_shape.type())
        {
            MIGRAPHX_THROW("Attention: Input and weight datatype must be the same");
        }

        if(weights_not_equal(weight_shape))
        {
            if(qkv_size_not_set(attr_out.qkv_hidden_sizes))
                MIGRAPHX_THROW("Attention: QKV size attribute must be set with non even weights");

            if(infered_out.has_past_input)
                MIGRAPHX_THROW("Attention: QKV size must be equally sized when using past/present buffers");
        }
        else
        {   
            // QKV is identical when second weight dim is divisible by 3 and qkv not set
            if(qkv_size_not_set(attr_out.qkv_hidden_sizes))
                attr_out.qkv_hidden_sizes = std::vector(3, (weight_shape.lens().at(1) / 3));
        }

        // Ensure qkv_hidden sizes set are valid wrt to input weights
        qkv_sizes_sum_arg_valid(attr_out.qkv_hidden_sizes, weight_tensor, 1, "weights");

        output_arg_vec.push_back(weight_tensor);
    }

    static void handle_projection_bias(const std::vector<instruction_ref>& args,
                                       struct attention_attr& attr_out,
                                       struct attention_infered& infered_out,
                                       std::vector<instruction_ref>& output_arg_vec)
    {
        instruction_ref bias;
        if(check_and_return_arg(args, 2, bias))
        {   
            auto bias_shape = bias->get_shape();
            auto bias_lens = bias_shape.lens();
            //ensure qkv dimension sum matches that of the bias vec
            qkv_sizes_sum_arg_valid(attr_out.qkv_hidden_sizes, bias, 0, "bias");
            if(args.at(0)->get_shape().type() != bias_shape.type())
            {
                MIGRAPHX_THROW("Attention: input bias must be the same type as input vector");
            }
            if(bias_lens.size() != 1)
            {
                MIGRAPHX_THROW("Attention: Bias requires tensor of (hidden_size + hidden_size + v_hidden_size) ");
            }
            output_arg_vec.push_back(bias);
            infered_out.has_input_bias = true;
        }
    }

    static void check_mask_index_shapes(const std::vector<size_t>& mask_index_lens,
                                         struct attention_infered& infered_out)
    {
        // Mask index is handled differently based on size of the input.
        //
        // raw attention mask has shape (batch, total sequence_length) 
        //                           or (batch, seq_length, total_sequence_length) with 0/1 values
        //                          where: total_sequence_length = sequence_length + past_sequence_length
        //
        // Right side padded has shape (batch) - value is sequence_length excluding padding
        // Left side Padding has shape (2 * batch) with inclusive start and exclusive end positions
        if(mask_index_lens.size() == 1)
        { // check left or right padding case
            if(mask_index_lens.at(0) == infered_out.batch_size)
            {
                infered_out.index_pad = RIGHT_PADDING;
            }
            else if(mask_index_lens.at(0) == (infered_out.batch_size * 2))
            {
                infered_out.index_pad = LEFT_PADDING;
            }
            else
            {
                MIGRAPHX_THROW("Attention: Invalid Mask_Index padding shape\n \
                                Use (batch) for Right Pad \n OR (batch *2) for Left Pad modes)");
            }
        }
        else if(mask_index_lens.size() == 2)
        { // This case assumes potentially past is set which is captured in total_sequence_length
            if(mask_index_lens.at(0) != infered_out.batch_size or 
               mask_index_lens.at(1) != infered_out.total_sequence_length)
            {
                MIGRAPHX_THROW("Attention: Invalid Mask_Index shape\n \
                                Use (batch, total_sequence_length) for shapes of size 2");
            }
            infered_out.index_pad = RAW;
        }
        else if(mask_index_lens.size() == 3)
        { // Similar to case 2 but with sequence length in dim 1
            if(mask_index_lens.at(0) != infered_out.batch_size or 
               mask_index_lens.at(1) != infered_out.sequence_length or
               mask_index_lens.at(2) != infered_out.total_sequence_length)
            {
                MIGRAPHX_THROW("Attention: Invalid Mask_Index shape\n \
                                Use (batch, sequence_length, total_sequence_length) for shapes of size 3");
            }
            infered_out.index_pad = RAW;
        }
        else if(mask_index_lens.size() == 4)
        { // Oddball case and can be used to infer max_sequence_length_parameter
            if(mask_index_lens.at(0) != infered_out.batch_size or 
               mask_index_lens.at(1) != 1 or 
               mask_index_lens.at(2) != mask_index_lens.at(3))
            {
                MIGRAPHX_THROW("Attention: Invalid Mask_Index shape\n  \
                                Use (batch, 1, max_sequence_length, max_sequence_length) for shapes of size 4");
            }
            infered_out.max_sequence_length = mask_index_lens.at(2);
            infered_out.index_pad = RAW;
        }
        else
        {
            MIGRAPHX_THROW("Attention: Mask_index Require shape of size either 1, 2, 3, 4 dimensions");
        }
    }

    static void handle_mask_index(const std::vector<instruction_ref>& args,
                                  struct attention_attr& attr_out,
                                  struct attention_infered& infered_out,
                                  std::vector<instruction_ref>& output_arg_vec)
    {
        instruction_ref mask_index;
        if(check_and_return_arg(args, 3, mask_index))
        {
            auto mask_index_shape = mask_index->get_shape();
            auto mask_index_lens  = mask_index_shape.lens();

            if(mask_index_shape.type() != migraphx::shape::int32_type)
            {
                MIGRAPHX_THROW("Attention: Mask_Index type must be int32 type");
            }
            check_mask_index_shapes(mask_index_lens, infered_out);
            infered_out.has_attn_mask = true;
            output_arg_vec.push_back(mask_index);
        }
    }

    static void handle_past(const std::vector<instruction_ref>& args,
                            struct attention_attr& attr_out,
                            struct attention_infered& infered_out,
                            std::vector<instruction_ref>& output_arg_vec)
    {
        instruction_ref past;
        if(check_and_return_arg(args, 4, past))
        {
            infered_out.has_past_input = true;
            if(args.size() != 7)
            {
                MIGRAPHX_THROW("Attention: Past input requires past_sequence_length to be set");
            }

            auto past_shape = past->get_shape();
            auto past_lens  = past_shape.lens();
            if((past_lens.at(0) != infered_out.batch_size) or 
               (past_lens.at(1) != attr_out.num_heads) or
               (past_lens.at(3) != infered_out.query_size) or 
               (past_lens.size() != 4))
            {
                MIGRAPHX_THROW("Attention: Past shape must be (batch, num_heads, max_sequence_length, head_size)\n OR when past_present_share_buffer set (batch, num_heads, total_sequence, head_size)");
            }

            if(attr_out.past_present_share_buffer)
            {
                infered_out.total_sequence_length = past_lens.at(2);
            }
            else
            {
                if(args.at(3)->get_shape().lens().size() == 4 and
                   args.at(3)->get_shape().lens().at(3) != past_lens.at(2))
                {
                    MIGRAPHX_THROW("Attention: Past invalid max_sequence_length");
                }
                else
                {
                    infered_out.max_sequence_length = past_lens.at(2);
                }
            }
            output_arg_vec.push_back(past);
        }
    }
    
    static void handle_attention_bias(const std::vector<instruction_ref>& args,
                                      struct attention_attr& attr_out,
                                      struct attention_infered& infered_out,
                                      std::vector<instruction_ref>& output_arg_vec)
    {
        instruction_ref attention_bias;
        if(check_and_return_arg(args, 5, attention_bias))
        {
            auto attn_bias_shape = attention_bias->get_shape();
            auto attn_bias_lens  = attn_bias_shape.lens();

            if(attn_bias_shape.type() != args.at(0)->get_shape().type())
            {
                MIGRAPHX_THROW("Attention: attention_bias must be same datatype as input tensor");
            }

            if((attn_bias_lens.at(0) != 1 and attn_bias_lens.at(0) != infered_out.batch_size) or
               (attn_bias_lens.at(1) != 1 and attn_bias_lens.at(1) != attr_out.num_heads) or
               (attn_bias_lens.at(2) != infered_out.sequence_length) or
               (attn_bias_lens.at(3) != infered_out.total_sequence_length))
            {
                MIGRAPHX_THROW("Attention: attention_bias shape must be (1 or Batch_size, 1 or num_heads, sequence_length, total_sequence_length");
            }
            infered_out.has_attn_bias = true;
            output_arg_vec.push_back(attention_bias);
        }
    }

    static void handle_past_sequence_length(const std::vector<instruction_ref>& args,
                                            struct attention_attr& attr_out,
                                            struct attention_infered& infered_out,
                                            std::vector<instruction_ref>& output_arg_vec)
    {
        instruction_ref past_seq_length;
        if(check_and_return_arg(args, 6, past_seq_length))
        {
            if(past_seq_length->get_shape().type() != shape::int32_type)
            {
                MIGRAPHX_THROW("past_sequence_length must be of type int32");
            }
            output_arg_vec.push_back(past_seq_length);
        }
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

        handle_input(args.at(0), attr_out, infered_out, input_arguments);
        handle_weight(args.at(1), args.at(0), attr_out, infered_out, input_arguments);

        // Handle theses individually. Order matters here to check conditions
        handle_projection_bias(args, attr_out, infered_out, input_arguments);
        handle_mask_index(args, attr_out, infered_out, input_arguments);
        handle_past(args, attr_out, infered_out, input_arguments);
        handle_attention_bias(args, attr_out, infered_out, input_arguments);
        handle_past_sequence_length(args, attr_out, infered_out, input_arguments);

        return input_arguments;
    }

    static std::vector<instruction_ref> even_split(const onnx_parser::node_info& info, 
                                                   const instruction_ref& input_matrix,
                                                   const size_t axis,
                                                   const size_t num_heads,
                                                   const size_t query_size)
    {
        std::vector<instruction_ref> result;

        for(auto i = 0; i < num_heads; i++)
        {
            auto starts = i * query_size;
            auto ends   = starts + query_size;
            auto op     = make_op("slice", {{"axes", {axis}}, {"starts", {starts}}, {"ends", {ends}}});

            result.push_back(info.add_instruction(op, input_matrix));
        }
        return result;
    }

    static std::vector<std::vector<instruction_ref>> qkv_split_per_head(const onnx_parser::node_info& info,
                                                           const std::vector<instruction_ref>& qkv_mats,
                                                           const attention_attr& attr_in,
                                                           const attention_infered& infered_in)
    {
        auto num_heads  = attr_in.num_heads;
        auto query_size = infered_in.query_size;

        auto split_q    = even_split(info, qkv_mats.at(0), 1, num_heads, query_size);
        auto split_k    = even_split(info, qkv_mats.at(1), 1, num_heads, query_size);
        auto split_v    = even_split(info, qkv_mats.at(2), 1, num_heads, query_size);

        std::vector<std::vector<instruction_ref>> qkv_split;
        for (size_t i = 0; i < num_heads; i++)
        {
            qkv_split.push_back({split_q.at(i), split_k.at(i), split_v.at(i)});
        }

        return qkv_split;
    } 

    static instruction_ref scale_dot_attention_head(const onnx_parser::node_info& info,
                                                    const std::vector<instruction_ref>& QKV,
                                                    const instruction_ref& scale_factor,
                                                    const instruction_ref& mask,
                                                    const instruction_ref& bias,
                                                    bool masked=false,
                                                    bool attn_bias=false)
    {
        auto Q = QKV.at(0);
        auto K = QKV.at(1);
        auto V = QKV.at(2);

        auto k_trans = info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1}}}), K);
        auto qk_out = info.add_instruction(make_op("dot"), Q, k_trans);
        auto qk_scaled = info.add_common_op("div", qk_out, scale_factor);

        auto qk_masked = qk_scaled;

        if(masked)
            qk_masked = info.add_common_op("add", qk_scaled, mask);

        auto qk_biased = qk_masked;
        if(attn_bias)
            qk_biased = info.add_common_op("add", qk_masked, bias);

        auto softmax_out = info.add_instruction(make_op("softmax"), qk_biased);
        auto output = info.add_instruction(make_op("dot"), softmax_out, V);
        output = info.add_instruction(make_op("transpose", {{"permutation", {1, 0, 2}}}), output);
        return output;
    }

    // Get Q, K, V matricies from stacked weight matrix
    static std::vector<instruction_ref> input_linear_to_qkv(const onnx_parser::node_info& info,
                                                            const instruction_ref& input,
                                                            const instruction_ref& stacked_weights,
                                                            const std::vector<size_t>& qkv_sizes,
                                                            const instruction_ref& input_bias,
                                                            const bool has_input_bias)
    {
        // Input encodes the batch, sequence_length and input_hidden_size (also known as embedding size) 
        auto input_lens = input->get_shape().lens();

        auto stacked_weights_unsq = info.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), stacked_weights);

        // Input stacked weights are (input_hidden_size, hidden_size + hidden_size + v_hidden_size) so slice out parts for each matrix
        // Since we known the input_hidden size is one dimension wee need to slice out the weight tensors accordingly before we perform matmul
        auto q_weight = info.add_instruction(make_op("slice", {{"axes",{2}}, {"starts", {0}}, {"ends", {qkv_sizes.at(0)}}}), stacked_weights_unsq);
        auto k_weight = info.add_instruction(make_op("slice", {{"axes",{2}}, {"starts", {qkv_sizes.at(0)}}, {"ends", {qkv_sizes.at(1) + qkv_sizes.at(0)}}}), stacked_weights_unsq);
        auto v_weight = info.add_instruction(make_op("slice", {{"axes",{2}}, {"starts", {qkv_sizes.at(0) + qkv_sizes.at(1)}}, {"ends", {qkv_sizes.at(0) + qkv_sizes.at(1) + qkv_sizes.at(2)}}}), stacked_weights_unsq);

        // Add in batch dimension to weights
        auto qk_lens = q_weight->get_shape().lens(); 
        qk_lens.at(0) = input_lens.at(0);
        auto v_lens = v_weight->get_shape().lens();
        v_lens.at(0) = input_lens.at(0);

        //Broadcast to batch size
        auto q_weight_bcast = info.add_instruction(make_op("multibroadcast", {{"out_lens", qk_lens}}), q_weight);
        auto k_weight_bcast = info.add_instruction(make_op("multibroadcast", {{"out_lens", qk_lens}}), k_weight);
        auto v_weight_bcast = info.add_instruction(make_op("multibroadcast", {{"out_lens", v_lens}}), v_weight);

        // Broadcast by batch then multiply
        auto Q = info.add_instruction(make_op("dot"), input, q_weight_bcast);
        auto K = info.add_instruction(make_op("dot"), input, k_weight_bcast);
        auto V = info.add_instruction(make_op("dot"), input, v_weight_bcast);

        std::vector<instruction_ref>qkv_mats{Q, K, V};
        return qkv_mats;
    }

    static instruction_ref create_input_mask(const onnx_parser::node_info& info,
                                             const instruction_ref& input,
                                             const instruction_ref& mask_input,
                                             const attention_infered& infered_in,
                                             const attention_attr& parsed_in)
    {
        instruction_ref final_mask;
        auto mask_value_literal = info.add_literal(migraphx::literal{migraphx::shape{input->get_shape().type(), {1}, {0}}, {parsed_in.mask_filter_val}});

        if(parsed_in.unidirectional)
        {
            // TODO Generate lower triangular mask and fill with mask_value_literal
        }
        else
        {
            if(infered_in.index_pad == RAW)
            {   // Raw case, 0 means mask 1 means pass through thus invert the matrix then multiply by mask_value
                auto bc_in = info.add_instruction(make_op("multibroadcast", {{"out_lens", mask_input->get_shape().lens()}}), mask_value_literal);
                final_mask = info.add_instruction(make_op("not"), mask_input);
                final_mask = info.add_instruction(make_op("convert", {{"target_type", input->get_shape().type()}}), final_mask);
                final_mask = info.add_instruction(make_op("mul"), final_mask, bc_in);
            }
            else if(infered_in.index_pad == LEFT_PADDING)
            {
                // TODO add left padding input case
            }
            else if(infered_in.index_pad == RIGHT_PADDING)
            {
                // TODO add right padding input case
            }
        }
        return final_mask;
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

        auto input_data = inputs.at(0);
        auto weights    = inputs.at(1);

        // Set projection bias when parsed in
        instruction_ref input_bias;
        if(infered_attributes.has_input_bias)
            input_bias = inputs.at(2);

        // Apply linear stage to QKV mats from weight matrix
        auto qkv_mats = input_linear_to_qkv(info, input_data, weights, parsed_attributes.qkv_hidden_sizes, input_bias, infered_attributes.has_input_bias);

        // Set attention mask and bias when detected on input
        instruction_ref attn_mask;
        if(infered_attributes.has_attn_mask)
            attn_mask = create_input_mask(info, input_data, inputs.at(3), infered_attributes, parsed_attributes);
        
        instruction_ref attn_bias;
        if(infered_attributes.has_attn_bias)
            attn_bias = inputs.at(5);

        auto attn_scale_factor = std::sqrt(infered_attributes.query_size);
        if(infered_attributes.scale_not_query_sz)
            attn_scale_factor = parsed_attributes.scale;

        // Used to scale all key values before any masking or other inputs
        auto scale_factor = info.add_literal(migraphx::literal{migraphx::shape{qkv_mats.at(0)->get_shape().type()},
                                                              {attn_scale_factor}});

        instruction_ref output;
        //Get vector of attention heads and then concat the output results
        if(parsed_attributes.num_heads > 1)
        {
            // Apply multi head splitting of qkv matrix prior to calculation
            auto split_qkv  = qkv_split_per_head(info, qkv_mats, parsed_attributes, infered_attributes);

            std::vector<instruction_ref> split_mask(parsed_attributes.num_heads);
            if(infered_attributes.has_attn_mask)
                split_mask = even_split(info, attn_mask, 1, parsed_attributes.num_heads, infered_attributes.query_size);

            std::vector<instruction_ref> split_bias(parsed_attributes.num_heads);
            if(infered_attributes.has_attn_bias)
                split_bias = even_split(info, attn_bias, 3, parsed_attributes.num_heads, infered_attributes.query_size);

            std::vector<instruction_ref> vec_of_attn_outs;
            std::transform(split_qkv.cbegin(),
                           split_qkv.cend(),
                           std::back_inserter(vec_of_attn_outs),
                           [&](auto && split_inputs) {
                            static size_t i = 0;
                            auto result = scale_dot_attention_head(info, split_inputs, scale_factor, split_mask.at(i), split_bias.at(i), infered_attributes.has_attn_mask, infered_attributes.has_attn_bias);
                            i = (i + 1) % parsed_attributes.num_heads;
                            return result;
                           });
            output = info.add_instruction(make_op("concat"), vec_of_attn_outs);
            output = info.add_instruction(make_op("transpose", {{"permutation", {1, 0, 2}}}), output);
        }
        else 
        {
            output = scale_dot_attention_head(info, qkv_mats, scale_factor, attn_mask, attn_bias, infered_attributes.has_attn_mask, infered_attributes.has_attn_bias);
        }

        std::vector<instruction_ref> output_vec{};
        output_vec.push_back(output);

        instruction_ref present;
        if(parsed_attributes.past_present_share_buffer)
        {
            present = output;
        }

        // Past and Present vetors must be used for the run.
        if(infered_attributes.has_past_input)
            output_vec.push_back(present);


        return output_vec;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
