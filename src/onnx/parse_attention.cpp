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

    enum class mask_pad
    {                  // Used to check mask_index when input vector size == 2
        no_pad,        // Not Set - If input isn't set indicates this is an error with op
        raw,           // Indicates input mask is raw mask where 0 masks the value and 1 does not.
        right_padding, // second dimension is (batch_size)
        left_padding,  // second dimension
    };

    enum class atten_mode
    {                    // Important to determine how to calculate total_sequence_length
        not_set,         // Not Set - If Past/Present used this indicates an error with op
        self_attention,  // Implies K,V lengths equal and equal to sequence_length
        cross_attention, // Relevant as K/V may have different lengths in this case
    };

    // For items explicitly parsed from attributes
    struct attention_attr
    {
        bool do_rotary = false; // Rotary encode input prior to projection with weight matrices
        bool past_present_share_buffer =
            false; // Related to past input shares buffer with present output
        bool unidirectional =
            false; // Mask is lower triangular ie) only pay attention to prev words
        std::size_t num_heads            = 1; // Required by inputs
        std::size_t rotary_embedding_dim = 0; // Gets set to head_size when not set
        std::vector<std::size_t> qkv_hidden_sizes{
            0, 0, 0};      // Sets hidden sizes if not set defiend by input
        float scale = 0.0; // Default 1/sqrt(query_size) (query_size also known as head_size)
        float mask_filter_val = -10000.0f; // Default value used for masking our input encoding
    };

    // Values inferred from input vectors or attributes
    struct attention_inferred
    {
        std::size_t batch_size; // From input dim(1)
        std::size_t
            input_hidden_size;   // From input dim(3) - Also known as embeddng_size in literature
        std::size_t hidden_size; // From weights or qkv_hidden_sizes, related to q,k size which must
                                 // be equal
        std::size_t
            v_hidden_size; // From weights or qkv_hiddn_sizes - differs during cross attention_mode
        std::size_t
            sequence_length; // Pulled from input dim(1), must be consistent between other parms
        std::size_t max_sequence_length;   // Pulled from mask_index
        std::size_t total_sequence_length; // Pulled from past_seq_length + sequence_length
        std::size_t query_size;            // Also known as head_size. Derived via num_heads
        bool has_past_input =
            false; // Set to true when we have past input. Present output required when set
        bool has_input_bias = false; // Set to true when we have input_bias
        bool has_attn_mask  = false; // Set to true when we have an attention mask set
        bool has_attn_bias  = false; // Set to true when we have an attention bias input
        bool scale_not_query_sz =
            false; // Set when a different scale attribute from 1/sqrt(query_size)
        mask_pad index_pad =
            mask_pad::no_pad; // Used to track state of input projection mask padding
        atten_mode attn_type; // Used to determine the attention configuration
    };

    static void handle_qkv_hidden_size_attr(const onnx_parser& parser,
                                            const onnx_parser::node_info& info,
                                            attention_attr& attr_out)
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
            MIGRAPHX_THROW("Attention: q and k hidden sizes must be identitical!");
        }

        std::vector<size_t> qkv_vec{static_cast<size_t>(qkv_values[0]),
                                    static_cast<size_t>(qkv_values[1]),
                                    static_cast<size_t>(qkv_values[2])};
        if(std::any_of(qkv_vec.begin(), qkv_vec.end(), [](auto i) { return (i == 0) or (i < 0); }))
        {
            MIGRAPHX_THROW("PARSE_ATTENTION: qkv_hidden_sizes must be nonzero and valid");
        }

        attr_out.qkv_hidden_sizes = qkv_vec;
    }

    static std::tuple<attention_attr, attention_inferred>
    handle_attributes(const onnx_parser& parser, const onnx_parser::node_info& info)
    {
        attention_attr attr_out;
        attention_inferred inferred_out;
        if(contains(info.attributes, "do_rotary"))
        { // TODO: Add rotary embedding support
            attr_out.do_rotary =
                (1 == parser.parse_value(info.attributes.at("do_rotary")).at<int>());
            if(attr_out.do_rotary)
                MIGRAPHX_THROW("PARSE_ATTENTION: Rotary Embedding in Attention OP not supported");
        }

        if(contains(info.attributes, "mask_filter_value"))
        {
            attr_out.mask_filter_val =
                parser.parse_value(info.attributes.at("mask_filter_value")).at<float>();
        }

        if(contains(info.attributes, "num_heads"))
        {
            attr_out.num_heads =
                parser.parse_value(info.attributes.at("num_heads")).at<std::size_t>();
        }
        else
        {
            MIGRAPHX_THROW("PARSE_ATTENTION: num_heads attribute required");
        }

        if(contains(info.attributes, "past_present_share_buffer"))
        {
            attr_out.past_present_share_buffer =
                (1 ==
                 parser.parse_value(info.attributes.at("past_present_share_buffer")).at<size_t>());
        }

        if(contains(info.attributes, "qkv_hidden_sizes"))
        {
            handle_qkv_hidden_size_attr(parser, info, attr_out);
        }

        if(contains(info.attributes, "rotary_embedding_dim"))
        { // TODO: Add rotary embedding support -- parsed but not used right now
            auto rotary_embedding_dim =
                parser.parse_value(info.attributes.at("rotary_embedding_dim")).at<size_t>();

            if(rotary_embedding_dim != 32 and rotary_embedding_dim != 64 and
               rotary_embedding_dim != 128)
            {
                MIGRAPHX_THROW(
                    "PARSE_ATTENTION: rotary_embedding_dim must be either 32, 64, or 128");
            }

            if(not attr_out.do_rotary)
            {
                MIGRAPHX_THROW(
                    "PARSE_ATTENTION: rotary_embedding_dim must be used with do_rotary attribute");
            }
            attr_out.rotary_embedding_dim = rotary_embedding_dim;
        }

        if(contains(info.attributes, "scale"))
        {
            attr_out.scale = parser.parse_value(info.attributes.at("scale")).at<float>();
            inferred_out.scale_not_query_sz = true;
        }

        if(contains(info.attributes, "unidirectional"))
        {
            MIGRAPHX_THROW("PARSE_ATTENTION: unidirectional attr not supported");
        }

        return {attr_out, inferred_out};
    }

    // qkv values must be greater than zero to be "set"
    static bool qkv_size_not_set(std::vector<size_t>& qkv_vec)
    {
        return std::any_of(qkv_vec.begin(), qkv_vec.end(), [](auto i) { return i <= 0; });
    }

    static void qkv_sizes_sum_arg_valid(const std::vector<size_t>& qkv_vec,
                                        const instruction_ref input_arg,
                                        const size_t dim,
                                        const std::string& name)
    {
        if(std::accumulate(qkv_vec.begin(), qkv_vec.end(), size_t(0)) !=
           input_arg->get_shape().lens().at(dim))
        {
            MIGRAPHX_THROW("Attention: q k v hidden sizes sum must match " + name + " tensor " +
                           std::to_string(dim) + " dimension");
        }
    }

    static bool weights_not_equal(const shape& weight_shape)
    {
        return (weight_shape.lens().at(1) % 3 != 0);
    }

    // simple call to check if the arg index exists
    static std::optional<instruction_ref>
    check_and_return_arg(const std::vector<instruction_ref>& args, const size_t index)
    {
        if(args.size() > index)
        {
            return args.at(index);
        }
        return {};
    }

    static instruction_ref handle_input(const instruction_ref& input_arg,
                                        const attention_attr& parsed_in,
                                        attention_inferred& inferred_out)
    {
        auto input_tensor = input_arg;
        auto input_shape  = input_tensor->get_shape();

        inferred_out.batch_size        = input_shape.lens().at(0);
        inferred_out.sequence_length   = input_shape.lens().at(1);
        inferred_out.input_hidden_size = input_shape.lens().at(2);
        // Determine the query_size used to generate attention heads that operate on each Q, K, V
        // matrix.
        inferred_out.query_size            = inferred_out.input_hidden_size / parsed_in.num_heads;
        inferred_out.total_sequence_length = inferred_out.sequence_length;

        return input_tensor;
    }

    static instruction_ref handle_weight(const instruction_ref& weight_arg,
                                         const instruction_ref& input_arg,
                                         attention_attr& attr_out,
                                         const attention_inferred& inferred_out)
    {
        auto weight_tensor = weight_arg;
        auto weight_shape  = weight_tensor->get_shape();
        auto input_shape   = input_arg->get_shape();

        if(weight_shape.lens().at(0) != input_shape.lens().at(2))
        {
            MIGRAPHX_THROW(
                "Attention: Input hidden size must be the same for input and weight tensors");
        }

        if(weight_shape.type() != input_shape.type())
        {
            MIGRAPHX_THROW("Attention: Input and weight datatype must be the same");
        }

        if(weights_not_equal(weight_shape))
        {
            if(qkv_size_not_set(attr_out.qkv_hidden_sizes))
                MIGRAPHX_THROW("Attention: QKV size attribute must be set with non even weights");

            if(inferred_out.has_past_input)
                MIGRAPHX_THROW(
                    "Attention: QKV size must be equally sized when using past/present buffers");
        }
        else
        {
            // QKV is identical when second weight dim is divisible by 3 and qkv not set
            if(qkv_size_not_set(attr_out.qkv_hidden_sizes))
            {
                std::vector<size_t> default_qkv_sizes(3, (weight_shape.lens().at(1) / 3));
                attr_out.qkv_hidden_sizes = default_qkv_sizes;
            }
        }

        // Ensure qkv_hidden sizes set are valid wrt to input weights
        qkv_sizes_sum_arg_valid(attr_out.qkv_hidden_sizes, weight_tensor, 1, "weights");

        return weight_tensor;
    }

    static std::optional<instruction_ref>
    handle_projection_bias(const std::vector<instruction_ref>& args,
                           const attention_attr& attr_out,
                           attention_inferred& inferred_out)
    {
        if(auto bias = check_and_return_arg(args, 2))
        {
            auto bias_shape       = (*bias)->get_shape();
            const auto& bias_lens = bias_shape.lens();
            // ensure qkv dimension sum matches that of the bias vec
            qkv_sizes_sum_arg_valid(attr_out.qkv_hidden_sizes, *bias, 0, "bias");
            if(args.at(0)->get_shape().type() != bias_shape.type())
            {
                MIGRAPHX_THROW("Attention: input bias must be the same type as input vector");
            }
            if(bias_lens.size() != 1)
            {
                MIGRAPHX_THROW("Attention: Bias requires tensor of (hidden_size + hidden_size + "
                               "v_hidden_size) ");
            }
            inferred_out.has_input_bias = true;
            return bias;
        }
        return nullopt;
    }

    static void check_mask_index_shapes(const std::vector<size_t>& mask_index_lens,
                                        attention_inferred& inferred_out)
    {
        // Mask index is handled differently based on size of the input.
        //
        // raw attention mask has shape (batch, total sequence_length)
        //                           or (batch, seq_length, total_sequence_length) with 0/1 values
        //                          where: total_sequence_length = sequence_length +
        //                          past_sequence_length
        //
        // Right side padded has shape (batch) - value is sequence_length excluding padding
        // Left side Padding has shape (2 * batch) with inclusive start and exclusive end positions
        if(mask_index_lens.size() == 1)
        { // check left or right padding case
            MIGRAPHX_THROW("Attention: Left/Right Padding not currently supported");
        }
        else if(mask_index_lens.size() == 2)
        { // This case assumes potentially past is set which is captured in total_sequence_length
            if(mask_index_lens.at(0) != inferred_out.batch_size or
               mask_index_lens.at(1) != inferred_out.total_sequence_length)
            {
                MIGRAPHX_THROW("Attention: Invalid Mask_Index shape\n \
                                Use (batch, total_sequence_length) for shapes of size 2");
            }
            inferred_out.index_pad = mask_pad::raw;
        }
        else if(mask_index_lens.size() == 3)
        { // Similar to case 2 but with sequence length in dim 1
            if(mask_index_lens.at(0) != inferred_out.batch_size or
               mask_index_lens.at(1) != inferred_out.sequence_length or
               mask_index_lens.at(2) != inferred_out.total_sequence_length)
            {
                MIGRAPHX_THROW("Attention: Invalid Mask_Index shape\n \
                                Use (batch, sequence_length, total_sequence_length) for shapes of size 3");
            }
            inferred_out.index_pad = mask_pad::raw;
            MIGRAPHX_THROW("Attention: Mask_index 3D masking not supported");
        }
        else if(mask_index_lens.size() == 4)
        { // Oddball case and can be used to infer max_sequence_length_parameter
            if(mask_index_lens.at(0) != inferred_out.batch_size or mask_index_lens.at(1) != 1 or
               mask_index_lens.at(2) != mask_index_lens.at(3))
            {
                MIGRAPHX_THROW("Attention: Invalid Mask_Index shape\n  \
                                Use (batch, 1, max_sequence_length, max_sequence_length) for shapes of size 4");
            }
            inferred_out.max_sequence_length = mask_index_lens.at(2);
            inferred_out.index_pad           = mask_pad::raw;
            MIGRAPHX_THROW("Attention: Mask_index 4D Megatron masking not supported");
        }
        else
        {
            MIGRAPHX_THROW(
                "Attention: Mask_index Require shape of size either 1, 2, 3, 4 dimensions");
        }
    }

    static std::optional<instruction_ref>
    handle_mask_index(const std::vector<instruction_ref>& args, attention_inferred& inferred_out)
    {
        if(auto mask_index = check_and_return_arg(args, 3))
        {
            auto mask_index_shape       = (*mask_index)->get_shape();
            const auto& mask_index_lens = mask_index_shape.lens();

            if(mask_index_shape.type() != migraphx::shape::int32_type)
            {
                MIGRAPHX_THROW("Attention: Mask_Index type must be int32 type");
            }
            check_mask_index_shapes(mask_index_lens, inferred_out);
            inferred_out.has_attn_mask = true;
            return mask_index;
        }
        return nullopt;
    }

    static void handle_past(const std::vector<instruction_ref>& args)
    {
        if(auto past = check_and_return_arg(args, 4))
        {
            MIGRAPHX_THROW("Attention: Past Not supported");
        }
    }

    static void handle_attention_bias(const std::vector<instruction_ref>& args)
    {
        if(auto attention_bias = check_and_return_arg(args, 5))
        {
            MIGRAPHX_THROW("Attention: attention_bias Not supported");
        }
    }

    static void handle_past_sequence_length(const std::vector<instruction_ref>& args)
    {
        if(auto past_seq_length = check_and_return_arg(args, 6))
        {
            MIGRAPHX_THROW("PARSE_ATTENTION: past_sequence_length not supported");
        }
    }

    static std::vector<instruction_ref> handle_arguments(const onnx_parser& /*parser*/,
                                                         const std::vector<instruction_ref>& args,
                                                         attention_attr& attr_out,
                                                         attention_inferred& inferred_out)
    {
        if(args.size() < 2 or args.size() > 7)
        {
            MIGRAPHX_THROW("Attention: Wrong number of inputs provided");
        }

        std::vector<instruction_ref> input_arguments{
            handle_input(args.at(0), attr_out, inferred_out),
            handle_weight(args.at(1), args.at(0), attr_out, inferred_out)};

        // Handle theses individually. Order matters here to check conditions
        if(auto bias = handle_projection_bias(args, attr_out, inferred_out))
            input_arguments.push_back(bias.value());

        if(auto mask = handle_mask_index(args, inferred_out))
            input_arguments.push_back(mask.value());

        // Currently not supported
        handle_past(args);
        handle_attention_bias(args);
        handle_past_sequence_length(args);
        return input_arguments;
    }

    static std::vector<instruction_ref>
    qkv_split_per_head(const onnx_parser::node_info& info,
                       const std::vector<instruction_ref>& qkv_mats,
                       const attention_attr& attr_in)
    {
        auto num_heads = attr_in.num_heads;
        auto q_lens    = qkv_mats.at(0)->get_shape().lens();
        auto k_lens    = qkv_mats.at(1)->get_shape().lens();
        auto v_lens    = qkv_mats.at(2)->get_shape().lens();

        // Split embedding into querry size and num heads from embedding dimension
        // Permute so we now result in (batch, sequence_length, querry_size, num_heads) prior to
        // calculations
        auto split_q = info.add_instruction(
            make_op("reshape",
                    {{"dims", {q_lens.at(0), q_lens.at(1), num_heads, q_lens.at(2) / num_heads}}}),
            qkv_mats.at(0));
        auto split_k = info.add_instruction(
            make_op("reshape",
                    {{"dims", {k_lens.at(0), k_lens.at(1), num_heads, k_lens.at(2) / num_heads}}}),
            qkv_mats.at(1));
        auto split_v = info.add_instruction(
            make_op("reshape",
                    {{"dims", {v_lens.at(0), v_lens.at(1), num_heads, v_lens.at(2) / num_heads}}}),
            qkv_mats.at(2));

        // Permute so we now result in (batch, num heads, sequence_length, querry_size) prior to
        // calculations
        split_q =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), split_q);
        split_k =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), split_k);
        split_v =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), split_v);

        return {split_q, split_k, split_v};
    }

    static instruction_ref scale_dot_attention_head(const onnx_parser::node_info& info,
                                                    const std::vector<instruction_ref>& qkv,
                                                    const instruction_ref& scale_factor,
                                                    const std::optional<instruction_ref>& mask,
                                                    const std::optional<instruction_ref>& bias)
    {
        auto q = qkv.at(0);
        auto k = qkv.at(1);
        auto v = qkv.at(2);

        auto k_trans =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 1, 3, 2}}}), k);
        auto qk_out = info.add_instruction(make_op("dot"), q, k_trans);

        // Apply bias to QK result
        auto qk_biased = qk_out;
        if(bias.has_value())
        {
            auto bc_bias = info.add_instruction(
                make_op("multibroadcast", {{"out_lens", qk_out->get_shape().lens()}}),
                bias.value());
            qk_biased = info.add_common_op("add", qk_out, bc_bias);
        }

        // Mask must be done after all bias and calculations done
        auto qk_masked = qk_biased;
        if(mask.has_value())
        {
            qk_masked = info.add_common_op("add", qk_masked, mask.value());
        }

        // Apply scale only after all the masking and biasing has occurred
        auto qk_scaled = info.add_common_op("mul", qk_masked, scale_factor);

        auto softmax_out = info.add_instruction(make_op("softmax", {{"axis", 3}}), qk_scaled);

        // Final result to compare with respect to values matrix
        auto output = info.add_instruction(make_op("dot"), softmax_out, v);

        // Transpose result from (batch, num heads, sequence_length, query_size) to (batch,
        // sequence_length, num_heads, query_size)
        output =
            info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), output);

        // Collapse back to (batch, sequence_length, query_size)
        auto lens = output->get_shape().lens();
        output    = info.add_instruction(
            make_op("reshape", {{"dims", {lens.at(0), lens.at(1), lens.at(2) * lens.at(3)}}}),
            output);
        return output;
    }

    // Get Q, K, V matricies from stacked weight matrix
    static std::vector<instruction_ref>
    input_linear_to_qkv(const onnx_parser::node_info& info,
                        const instruction_ref& input,
                        const instruction_ref& stacked_weights,
                        const std::vector<size_t>& qkv_sizes,
                        const std::optional<instruction_ref>& input_bias)
    {
        // Input encodes the batch, sequence_length and input_hidden_size (also known as embedding
        // size)
        auto input_lens = input->get_shape().lens();

        auto stacked_weights_unsq =
            info.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), stacked_weights);
        auto w_lens                     = stacked_weights_unsq->get_shape().lens();
        w_lens.at(0)                    = input_lens.at(0);
        auto stacked_weights_unsq_bcast = info.add_instruction(
            make_op("multibroadcast", {{"out_lens", w_lens}}), stacked_weights_unsq);

        auto stacked_result =
            info.add_instruction(make_op("dot"), input, stacked_weights_unsq_bcast);

        if(input_bias.has_value())
        {
            stacked_result = info.add_common_op("add", stacked_result, input_bias.value());
        }

        // Input stacked weights are (input_hidden_size, hidden_size + hidden_size + v_hidden_size)
        // so slice out parts for each matrix Since we known the input_hidden size is one dimension
        // wee need to slice out the weight tensors accordingly before we perform matmul
        auto q = info.add_instruction(
            make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {qkv_sizes.at(0)}}}),
            stacked_result);
        auto k = info.add_instruction(make_op("slice",
                                              {{"axes", {2}},
                                               {"starts", {qkv_sizes.at(0)}},
                                               {"ends", {qkv_sizes.at(1) + qkv_sizes.at(0)}}}),
                                      stacked_result);
        auto v = info.add_instruction(
            make_op("slice",
                    {{"axes", {2}},
                     {"starts", {qkv_sizes.at(0) + qkv_sizes.at(1)}},
                     {"ends", {qkv_sizes.at(0) + qkv_sizes.at(1) + qkv_sizes.at(2)}}}),
            stacked_result);

        std::vector<instruction_ref> qkv_mats{q, k, v};
        return qkv_mats;
    }

    // Slice, mul, convert and concat until we get a mask matrix useful prior to the where
    static instruction_ref generate_raw_mask_per_batch(const onnx_parser::node_info& info,
                                                       const instruction_ref& input,
                                                       const instruction_ref& mask_input,
                                                       const attention_attr& parsed_in,
                                                       const attention_inferred& inferred_in)
    {
        auto batch_size    = inferred_in.batch_size;
        auto total_seq_len = inferred_in.total_sequence_length;
        auto num_heads     = parsed_in.num_heads;

        // Other two cases require us to generate masks from sequence or total sequence length pads.
        auto pass_value_lit = info.add_literal(
            migraphx::literal{migraphx::shape{input->get_shape().type(), {1}, {1}}, {0}});
        auto mask_value_lit = info.add_literal(migraphx::literal{
            migraphx::shape{input->get_shape().type(), {1}, {1}}, {parsed_in.mask_filter_val}});

        // For dim = 2 or dim =3 generate the apporiate mask across batches
        // We need to handle the batch case since raw masking involes shape [batch, seq_len] or
        // [batch, seq_len, total_seq_len],
        auto bc_pass = info.add_instruction(
            make_op("multibroadcast",
                    {{"out_lens", {batch_size, num_heads, total_seq_len, total_seq_len}}}),
            pass_value_lit);
        auto bc_mask = info.add_instruction(
            make_op("multibroadcast",
                    {{"out_lens", {batch_size, num_heads, total_seq_len, total_seq_len}}}),
            mask_value_lit);

        auto raw_mask = mask_input;
        // For raw masks we just need to mask out key value padding thus the 3d mask isn't needed
        // here.
        raw_mask = info.add_instruction(
            make_op("reshape", {{"dims", {batch_size, 1, 1, total_seq_len}}}), raw_mask);
        raw_mask = info.add_instruction(
            make_op("multibroadcast",
                    {{"out_lens", {batch_size, num_heads, total_seq_len, total_seq_len}}}),
            raw_mask);
        raw_mask = info.add_instruction(
            make_op("reshape", {{"dims", {batch_size, num_heads, total_seq_len, total_seq_len}}}),
            raw_mask);

        // Reuse "0" broadcasted converted to int32 to check if input mask is greater than 0 for
        // where condition
        auto in_pass = info.add_instruction(
            make_op("convert", {{"target_type", mask_input->get_shape().type()}}), bc_pass);
        auto in_bool = info.add_instruction(make_op("equal"), raw_mask, in_pass);
        // Need this to let MLIR to run with where
        in_bool = info.add_instruction(
            make_op("convert", {{"target_type", migraphx::shape::int8_type}}), in_bool);
        return info.add_instruction(make_op("where"), in_bool, bc_mask, bc_pass);
    }

    static std::optional<instruction_ref>
    create_input_mask(const onnx_parser::node_info& info,
                      const instruction_ref& input, // TODO: Convert this to type
                      const instruction_ref& mask_input,
                      const attention_inferred& inferred_in,
                      const attention_attr& parsed_in)
    {
        // Shape Scale dot attention prior to mask will be in (batch, num_heads, query_size,
        // query_size) thus mask needs to handle batch and query_size We should return mask of
        // batch, 1, query_size, query_size so that this per-batch masked can be broadcasted across
        // each attention head

        if(inferred_in.index_pad == mask_pad::raw)
        { // Raw Mask - 0 means mask, 1 means pass through. Apply mask_filter_val to mask indicies
          // and zero otherwise
            // Need to generate from 2 dims or 3 dim cases
            return generate_raw_mask_per_batch(info, input, mask_input, parsed_in, inferred_in);
        }

        return nullopt;
    }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        auto [parsed_attributes, inferred_attributes] = handle_attributes(parser, info);
        auto inputs = handle_arguments(parser, args, parsed_attributes, inferred_attributes);

        auto input_data = inputs.at(0);
        auto weights    = inputs.at(1);
        // Set projection bias when parsed in
        std::optional<instruction_ref> input_bias;
        if(inferred_attributes.has_input_bias)
            input_bias = inputs.at(2);

        // Apply linear stage to QKV mats from weight matrix - If past input just return q mat later
        // split will be extracted from past vector
        auto qkv_mats = input_linear_to_qkv(
            info, input_data, weights, parsed_attributes.qkv_hidden_sizes, input_bias);

        // Set attention mask and bias when detected on input
        std::optional<instruction_ref> attn_mask;
        if(inferred_attributes.has_attn_mask)
            attn_mask = create_input_mask(
                info, input_data, inputs.at(3), inferred_attributes, parsed_attributes);

        std::optional<instruction_ref> attn_bias;
        if(inferred_attributes.has_attn_bias)
            attn_bias = inputs.at(5);

        float attn_scale_factor = inferred_attributes.query_size;
        if(inferred_attributes.scale_not_query_sz)
            attn_scale_factor = parsed_attributes.scale;

        // Used to scale all key values before any masking or other inputs
        auto scale_factor = info.add_literal(migraphx::literal{
            migraphx::shape{qkv_mats.at(0)->get_shape().type()}, {attn_scale_factor}});

        if(not inferred_attributes.scale_not_query_sz)
        {
            scale_factor = info.add_instruction(make_op("sqrt"), scale_factor);
            scale_factor = info.add_instruction(make_op("recip"), scale_factor);
        }

        // split QKV into proper batched attention head shape before we perform scale_dot_attention
        // (saves us a concat)
        auto split_qkv = qkv_split_per_head(info, qkv_mats, parsed_attributes);

        return scale_dot_attention_head(info, split_qkv, scale_factor, attn_mask, attn_bias);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
