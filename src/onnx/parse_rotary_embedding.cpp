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
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct rotary_parameters
{
    // Extracted from Inputs
    std::size_t batch_size  = 0; // Batch used by input
    std::size_t seq_len     = 0; // Sequence length used by input
    std::size_t head_size   = 0; // Head size used for offset in each block
    std::size_t max_seq_len = 0; // Sequence length used by sin/cos caches
    std::size_t hidden_size = 0; // Hidden size used by the input
    // input shape:
    // true => [batch_size, num_heads, seq_len, head_size]
    // false => [batch_size, seq_len, hidden_size=num_heads*head_size]
    bool is_bnsh = false;
    std::vector<shape::dynamic_dimension> input_dims;

    // Extracted from both
    std::size_t num_heads = 0;     // num_heads = hidden_size / head_size  or input
    bool head_diff        = false; // head_size > rotary_embedding_dim

    // Extracted from Attributes
    std::size_t rotary_embedding_dim = 0;
    bool interleaved                 = false;
    bool is_packed_batching          = false;
    float scale                      = 1.0;
};

struct parse_rotary_embedding : op_parser<parse_rotary_embedding>
{
    std::vector<op_desc> operators() const { return {{"RotaryEmbedding"}}; }

    static std::size_t
    fixed_dimension(const shape& s, std::size_t axis, const std::string& name)
    {
        if(not s.dynamic())
            return s.lens().at(axis);
        auto result = sym::fixed_value(s.dyn_dims().at(axis).sym_expr);
        if(not result.has_value())
            MIGRAPHX_THROW("RotaryEmbedding: " + name + " must be fixed");
        return sym::to<std::size_t>(*result);
    }

    static instruction_ref add_runtime_reshape(const onnx_parser::node_info& info,
                                               instruction_ref input,
                                               const std::vector<shape::dynamic_dimension>& dims)
    {
        std::vector<sym::expr> expressions(dims.size());
        transform(dims, expressions.begin(), [](const auto& dim) { return dim.sym_expr; });
        auto resolved_dims = info.add_instruction(
            make_op("eval_expr_from_shape", {{"expressions", to_value(expressions)}}),
            info.mod->get_parameters());
        auto allocation = info.add_instruction(
            make_op("allocate", {{"shape", to_value(shape{input->get_shape().type(), dims})}}),
            resolved_dims);
        return info.add_instruction(make_op("reshape"), input, allocation);
    }

    static void parse_attributes(const onnx_parser& parser,
                                 const onnx_parser::node_info& info,
                                 rotary_parameters& param)

    {
        if(contains(info.attributes, "interleaved"))
        {
            param.interleaved = parser.parse_value(info.attributes.at("interleaved")).at<bool>();
        }

        if(contains(info.attributes, "is_packed_batching"))
        {
            // Ragged batching is a form of dynamic batching where inputs of varying lengths
            // are padded and batched together. E.g. inputs of [1, 3], [1, 4], [1, 5]
            // would each be padded to [1, 5] and batched together as [3, 5]
            param.is_packed_batching =
                parser.parse_value(info.attributes.at("is_packed_batching")).at<bool>();
            if(param.is_packed_batching)
            {
                MIGRAPHX_THROW(
                    "RotaryEmbedding: is_packed_batching aka ragged batching is not supported.");
            }
        }

        if(contains(info.attributes, "num_heads"))
        {
            param.num_heads = parser.parse_value(info.attributes.at("num_heads")).at<std::size_t>();
        }

        if(contains(info.attributes, "rotary_embedding_dim"))
        {
            param.rotary_embedding_dim =
                parser.parse_value(info.attributes.at("rotary_embedding_dim")).at<std::size_t>();
        }

        if(contains(info.attributes, "scale"))
        {
            // ContribOp spec does not specify how to apply the scale and ORT neither
            // handles nor tests this attribute
            param.scale = parser.parse_value(info.attributes.at("scale")).at<float>();
            if(not float_equal(param.scale, 1.0f))
            {
                MIGRAPHX_THROW("RotaryEmbedding: scale is not supported.");
            }
        }

        if((param.num_heads != 0) xor (param.rotary_embedding_dim != 0))
        {
            MIGRAPHX_THROW("RotaryEmbedding: num_heads and rotary_embedding dims must be used "
                           "together and non-zero");
        }
    }

    static void parse_input(const instruction_ref& input, rotary_parameters& param)
    {
        const auto& input_shape = input->get_shape();
        auto input_dims         = input_shape.ndim();

        if(input_dims < 3 or input_dims > 4)
        {
            MIGRAPHX_THROW(
                "RotaryEmbedding:Input must be 3D (Batch , Sequence Length, Hidden size) or \
                            4D (Batch, Num Heads, Sequence Length, Head Size))");
        }

        if(input_shape.symbolic())
        {
            param.input_dims = input_shape.dyn_dims();
            if(input_dims == 3)
            {
                param.hidden_size = fixed_dimension(input_shape, 2, "hidden size");
            }
            else
            {
                param.num_heads   = fixed_dimension(input_shape, 1, "number of heads");
                param.head_size   = fixed_dimension(input_shape, 3, "head size");
                param.hidden_size = param.num_heads * param.head_size;
                param.is_bnsh     = true;
            }
            return;
        }

        auto input_lens   = input_shape.lens();
        param.batch_size = input_lens.at(0);

        if(input_dims == 3)
        {
            param.seq_len     = input_lens.at(1);
            param.hidden_size = input_lens.at(2);
        }
        else
        {
            param.num_heads   = input_lens.at(1);
            param.seq_len     = input_lens.at(2);
            param.head_size   = input_lens.at(3);
            param.hidden_size = param.num_heads * param.head_size;
            param.is_bnsh     = true;
        }
    }

    // Ensure position ID shapes comply with input dimensions
    static void parse_position_ids(const instruction_ref& position_ids,
                                   const rotary_parameters& param)
    {
        const auto& position_shape = position_ids->get_shape();
        auto position_dim          = position_shape.ndim();

        if(position_dim > 2 or position_shape.scalar())
        {
            MIGRAPHX_THROW("RotaryEmbedding: Position_ids must be either 1D tensor of shape (1) or "
                           "2d (Batch, Sequence Length)");
        }

        if(not param.input_dims.empty())
        {
            if(position_dim != 2)
                MIGRAPHX_THROW(
                    "RotaryEmbedding: symbolic position_ids must have shape (Batch, Sequence)");
            auto position_dims = position_shape.to_symbolic().dyn_dims();
            auto sequence_axis = param.is_bnsh ? 2 : 1;
            if(not sym::same_symbol(position_dims.at(0).sym_expr,
                                    param.input_dims.at(0).sym_expr) or
               not sym::same_symbol(position_dims.at(1).sym_expr,
                                    param.input_dims.at(sequence_axis).sym_expr))
                MIGRAPHX_THROW("RotaryEmbedding: Position_id 2D dims must match input batch size "
                               "and sequence length");
            return;
        }

        auto position_len = position_shape.lens();
        if(position_dim == 1 and position_len.at(0) != 1)
        {
            MIGRAPHX_THROW("RotaryEmbedding: Position_id must have shape of 1 for 1D tensor");
        }

        if((position_dim == 2) and
           ((position_len.at(0) != param.batch_size) or (position_len.at(1) != param.seq_len)))
        {
            MIGRAPHX_THROW("RotaryEmbedding: Position_id 2D dims must match input batch size and "
                           "sequence length");
        }
    }

    static void parse_cos_cache(const instruction_ref& cos_cache, rotary_parameters& param)
    {
        const auto& cache_shape = cos_cache->get_shape();
        if(cache_shape.ndim() != 2)
            MIGRAPHX_THROW("RotaryEmbedding: cosine cache must be rank 2");
        param.max_seq_len = fixed_dimension(cache_shape, 0, "cosine cache sequence length");
        auto cache_width  = fixed_dimension(cache_shape, 1, "cosine cache width");
        if(param.num_heads == 0)
        {
            param.head_size = cache_width * 2;
            param.num_heads = param.hidden_size / param.head_size;
        }
        else
        {
            param.head_size = param.hidden_size / param.num_heads;
        }
        if(param.rotary_embedding_dim == 0)
        {
            param.rotary_embedding_dim = param.head_size;
        }
        else if(param.head_size > param.rotary_embedding_dim)
        {
            param.head_diff = true;
        }
        else if(param.head_size < param.rotary_embedding_dim)
        {
            MIGRAPHX_THROW("RotaryEmbedding: rotary_embedding_dim must be <= head_size");
        }

        compare_sin_cos_cache_dims(cache_width, param);
    }

    static void parse_sin_cache(const instruction_ref& sin_cache, const rotary_parameters& param)
    {
        const auto& cache_shape = sin_cache->get_shape();
        if(cache_shape.ndim() != 2)
            MIGRAPHX_THROW("RotaryEmbedding: sine cache must be rank 2");
        auto max_seq_len = fixed_dimension(cache_shape, 0, "sine cache sequence length");

        if(param.max_seq_len != max_seq_len)
        {
            MIGRAPHX_THROW(
                "RotaryEmbedding: max_sequence_length must be the same between sin & cos caches!");
        }

        compare_sin_cos_cache_dims(
            fixed_dimension(cache_shape, 1, "sine cache width"), param);
    }

    static void compare_sin_cos_cache_dims(const size_t dim, const rotary_parameters& param)
    {
        if(param.rotary_embedding_dim != 0 and param.rotary_embedding_dim / 2 != dim)
        {
            MIGRAPHX_THROW(
                "RotaryEmbedding: rotary_embedding must be the same between sin & cos caches!");
        }
    }

    static void parse_input_args(const std::vector<instruction_ref>& args, rotary_parameters& param)
    {
        // Order matters as we're basing params related to the first input
        parse_input(args.at(0), param);
        parse_position_ids(args.at(1), param);
        parse_cos_cache(args.at(2), param);
        parse_sin_cache(args.at(3), param);
    }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args) const
    {
        if(args.size() != 4)
        {
            MIGRAPHX_THROW("RotaryEmbedding: Wrong number of inputs provided require 4");
        }

        // Sanity check input dimension and shapes while extracting params
        rotary_parameters params{};
        parse_attributes(parser, info, params);
        parse_input_args(args, params);

        // Setup based on parsed params gathered from input attributes/inputs
        auto input        = args.at(0);
        auto position_ids = args.at(1);
        auto cos_cache    = args.at(2);
        auto sin_cache    = args.at(3);

        if(not params.is_bnsh)
        {
            if(params.input_dims.empty())
            {
                input = info.add_instruction(
                    make_op("reshape",
                            {{"dims",
                              {params.batch_size,
                               params.seq_len,
                               params.num_heads,
                               params.head_size}}}),
                    input);
            }
            else
            {
                auto dims = params.input_dims;
                dims.at(2) = shape::dynamic_dimension{sym::lit(params.num_heads)};
                dims.push_back(shape::dynamic_dimension{sym::lit(params.head_size)});
                input = add_runtime_reshape(info, input, dims);
            }
            input =
                info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), input);
        }
        instruction_ref tail;
        if(params.head_diff)
        {
            tail  = info.add_instruction(make_op("slice",
                                                 {{"axes", {-1}},
                                                  {"starts", {params.rotary_embedding_dim}},
                                                  {"ends", {params.head_size}}}),
                                        input);
            input = info.add_instruction(
                make_op("slice",
                        {{"axes", {-1}}, {"starts", {0}}, {"ends", {params.rotary_embedding_dim}}}),
                input);
        }

        auto output = op::builder::add("rotary_embedding",
                                       *info.mod,
                                       {input, position_ids, cos_cache, sin_cache},
                                       {{"interleaved", params.interleaved}})
                          .at(0);

        if(params.head_diff)
        {
            output = info.add_instruction(make_op("concat", {{"axis", -1}}), output, tail);
        }
        if(not params.is_bnsh)
        {
            output =
                info.add_instruction(make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), output);
            if(params.input_dims.empty())
            {
                output = info.add_instruction(
                    make_op("reshape",
                            {{"dims", {params.batch_size, params.seq_len, params.hidden_size}}}),
                    output);
            }
            else
            {
                output = add_runtime_reshape(info, output, params.input_dims);
            }
        }

        return {output};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
