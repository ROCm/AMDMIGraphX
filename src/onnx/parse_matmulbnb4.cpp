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
#include "migraphx/errors.hpp"
#include "migraphx/instruction_ref.hpp"
#include "migraphx/onnx/onnx_parser.hpp"
#include <cstddef>
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_matmulbnb4 : op_parser<parse_matmulbnb4>
{
    std::vector<op_desc> operators() const { return {{"MatMulBnb4"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        // Parse required attributes
        const size_t n          = parse_attribute(parser, info, "N");
        const size_t k          = parse_attribute(parser, info, "K");
        const size_t block_size = parse_attribute(parser, info, "block_size");
        
        // Parse quant_type attribute (0 for FP4, 1 for NF4)
        int quant_type = 0;  // default to FP4
        if(contains(info.attributes, "quant_type"))
        {
            quant_type = parse_attribute(parser, info, "quant_type");
        }
        
        // Validate quant_type
        if(quant_type != 0 && quant_type != 1)
        {
            MIGRAPHX_THROW("MatMulBnb4: quant_type must be 0 (FP4) or 1 (NF4), actual value: " + 
                           std::to_string(quant_type));
        }

        // Validate block_size (must be power of 2 and >= 16)
        if(block_size < 16 || (block_size & (block_size - 1)) != 0)
        {
            MIGRAPHX_THROW("MatMulBnb4: block_size must be a power of 2 and >= 16, actual value: " +
                           std::to_string(block_size));
        }

        // Validate inputs
        if(args.size() < 3)
        {
            MIGRAPHX_THROW("MatMulBnb4: requires exactly 3 inputs (A, B, absmax)");
        }

        // Validate Input A
        if(args[0]->get_shape().ndim() < 2)
        {
            MIGRAPHX_THROW("MatMulBnb4: Input A must have at least 2 dimensions");
        }
        
        auto a_inner_dim = args[0]->get_shape().lens().back();
        if(a_inner_dim != k)
        {
            MIGRAPHX_THROW("MatMulBnb4: Input A inner dimension (" + std::to_string(a_inner_dim) + 
                           ") must match attribute K (" + std::to_string(k) + ")");
        }

        // Validate Input B dimensions
        std::vector<size_t> expected_b_lens{(n * k + 1) / 2};
        if(args[1]->get_shape().lens() != expected_b_lens)
        {
            MIGRAPHX_THROW("MatMulBnb4: Input B does not match expected dims: " +
                           to_string_range(expected_b_lens) +
                           ". Actual dims: " + to_string_range(args[1]->get_shape().lens()));
        }

        // Validate Input absmax dimensions  
        const size_t expected_absmax_elements = (n * k + block_size - 1) / block_size;
        if(args[2]->get_shape().elements() != expected_absmax_elements)
        {
            MIGRAPHX_THROW("MatMulBnb4: Input absmax does not match expected dims: " +
                           std::to_string(expected_absmax_elements) +
                           ". Actual dims: " + to_string_range(args[2]->get_shape().lens()));
        }

        // Dequantize input B using the provided absmax scales
        auto dequantized_b = dequantize_b_bnb4(info, n, k, block_size, quant_type, args);
        dequantized_b = info.add_instruction(make_op("transpose", {{"permutation", {1, 0}}}), dequantized_b);
        
        // Perform the matrix multiplication
        return matmul(info, args[0], dequantized_b);
    }

private:
    int parse_attribute(const onnx_parser& parser,
                        onnx_parser::node_info& info,
                        const std::string& attribute_name) const
    {
        if(not contains(info.attributes, attribute_name))
        {
            MIGRAPHX_THROW("MatMulBnb4: Attribute " + attribute_name + " is required but missing");
        }

        return parser.parse_value(info.attributes[attribute_name]).at<int>();
    }

    instruction_ref dequantize_b_bnb4(onnx_parser::node_info& info,
                                     size_t n,
                                     size_t k,
                                     size_t block_size,
                                     int quant_type,
                                     const std::vector<instruction_ref>& args) const
    {        
        // Unpack the 4-bit quantized data
        auto unpacked_b = unpack_bnb4_data(info, n, k, args[1]);
        
        // Prepare absmax for blockwise dequantization
        auto prepared_absmax = prepare_blockwise_absmax(info, n, k, block_size, args[2]);
        
        // Apply dequantization based on quantization type
        return apply_bnb4_dequantization(info, unpacked_b, prepared_absmax, quant_type);
    }

    instruction_ref unpack_bnb4_data(onnx_parser::node_info& info,
                                    size_t n,
                                    size_t k,
                                    instruction_ref b) const
    {     
        // For BNB4, the input B is transposed, flattened and quantized blockwise
        // We need to unpack the 4-bit data first
        
        // Unpack the 4-bit data
        auto unpacked = info.add_instruction(make_op("unpack_int4"), b);
        
        // Reshape to (k, n) since B was transposed before quantization
        unpacked = info.add_instruction(make_op("reshape", {{"dims", {k, n}}}), unpacked);
        
        return unpacked;
    }

    instruction_ref prepare_blockwise_absmax(onnx_parser::node_info& info,
                                            size_t n,
                                            size_t k,
                                            size_t block_size,
                                            instruction_ref absmax) const
    {
        // absmax is a 1D tensor with (n * k + block_size - 1) / block_size elements
        // Each element represents the scale for one block
        
        // Expand absmax to apply blockwise scaling
        auto expanded_absmax = info.add_instruction(make_op("unsqueeze", {{"axes", {1}}}), absmax);
        
        auto bc_lens = expanded_absmax->get_shape().lens();
        bc_lens[1] = block_size;
        expanded_absmax = info.add_instruction(make_op("multibroadcast", {{"out_lens", bc_lens}}), expanded_absmax);
        
        // Flatten to get scales for all elements
        expanded_absmax = info.add_instruction(make_op("reshape", {{"dims", {n * k}}}), expanded_absmax);
        
        // Handle runt block if n*k is not divisible by block_size
        const size_t total_elements = n * k;
        if(expanded_absmax->get_shape().lens()[0] > total_elements)
        {
            expanded_absmax = info.add_instruction(
                make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {total_elements}}}), expanded_absmax);
        }
        
        // Reshape to (n, k) then transpose to (k, n) to match unpacked B tensor
        expanded_absmax = info.add_instruction(make_op("reshape", {{"dims", {n, k}}}), expanded_absmax);
        expanded_absmax = info.add_instruction(make_op("transpose", {{"permutation", {1, 0}}}), expanded_absmax);
        
        return expanded_absmax;
    }

    instruction_ref apply_bnb4_dequantization(onnx_parser::node_info& info,
                                             instruction_ref quantized_data,
                                             instruction_ref absmax,
                                             int quant_type) const
    {
        // Convert quantized data to float for dequantization
        auto float_data = info.add_instruction(
            make_op("convert", {{"target_type", migraphx::shape::float_type}}), quantized_data);
        
        // Apply BNB4 specific dequantization formula
        if(quant_type == 0) // FP4
        {
            // For FP4: dequantized = quantized * absmax / scale_factor
            // FP4 scale factor is typically 8 (since it represents values in [-8, 7] range)
            auto scale_factor = info.add_literal(
                migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {8.0f}});
            auto scale_factor_bc = info.add_instruction(
                make_op("multibroadcast", {{"out_lens", float_data->get_shape().lens()}}), scale_factor);
            
            auto scaled_data = info.add_instruction(make_op("div"), float_data, scale_factor_bc);
            return info.add_instruction(make_op("mul"), scaled_data, absmax);
        }
        else // NF4 (quant_type == 1)
        {
            // For NF4: more complex dequantization using lookup table approach
            // For now, implement a simplified version similar to FP4
            // In practice, NF4 uses a specific lookup table for dequantization
            auto scale_factor = info.add_literal(
                migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {8.0f}});
            auto scale_factor_bc = info.add_instruction(
                make_op("multibroadcast", {{"out_lens", float_data->get_shape().lens()}}), scale_factor);
            
            auto scaled_data = info.add_instruction(make_op("div"), float_data, scale_factor_bc);
            return info.add_instruction(make_op("mul"), scaled_data, absmax);
        }
    }

    instruction_ref matmul(onnx_parser::node_info& info, instruction_ref a, instruction_ref b) const
    {
        const auto a_rank = a->get_shape().ndim();
        
        if(a_rank == 1)
        {
            a = info.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), a);
        }
        
        // B should be 2D (k, n) after dequantization
        // If A has more than 2 dimensions, broadcast B accordingly
        if(a_rank > 2)
        {
            auto b_lens = b->get_shape().lens();
            auto b_bc_lens = a->get_shape().lens();
            // Set the last two dimensions to match B's dimensions
            std::copy(b_lens.begin(), b_lens.end(), b_bc_lens.end() - 2);
            b = info.add_instruction(make_op("multibroadcast", {{"out_lens", b_bc_lens}}), b);
        }

        auto dot = info.add_instruction(make_op("dot"), a, b);

        // Squeeze dimensions if they were added
        if(a_rank == 1)
        {
            dot = info.add_instruction(
                make_op("squeeze", {{"axes", {dot->get_shape().ndim() - 2}}}), dot);
        }

        return dot;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
