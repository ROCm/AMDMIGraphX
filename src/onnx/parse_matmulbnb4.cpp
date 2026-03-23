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
#include <migraphx/errors.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/onnx/onnx_parser.hpp>
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
        const size_t n          = parse_attribute(parser, info, "N");
        const size_t k          = parse_attribute(parser, info, "K");
        const size_t block_size = get_block_size(parser, info);
        const size_t quant_type = get_quant_type(parser, info);

        if(args.size() < 3)
        {
            MIGRAPHX_THROW("MatMulBnb4: requires exactly 3 inputs (A, B, absmax)");
        }

        check_a(args[0]);
        check_k(args[0], k);
        check_b(args[1], n, k);
        check_absmax(args[2], n, k, block_size);

        auto dequantized_b = dequantize_b_bnb4(info, n, k, block_size, quant_type, args);
        dequantized_b =
            info.add_instruction(make_op("transpose", {{"permutation", {1, 0}}}), dequantized_b);

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

    size_t get_quant_type(const onnx_parser& parser, onnx_parser::node_info& info) const
    {
        const size_t quant_type = parse_attribute(parser, info, "quant_type");
        
        if(quant_type != 0 and quant_type != 1)
        {
            MIGRAPHX_THROW("MatMulBnb4: quant_type must be 0 (FP4) or 1 (NF4), actual value: " +
                           std::to_string(quant_type));
        }
        
        return quant_type;
    }

    size_t get_block_size(const onnx_parser& parser, onnx_parser::node_info& info) const
    {
        const size_t block_size = parse_attribute(parser, info, "block_size");
        
        if(block_size < 16 or (block_size & (block_size - 1)) != 0)
        {
            MIGRAPHX_THROW("MatMulBnb4: block_size must be a power of 2 and >= 16, actual value: " +
                           std::to_string(block_size));
        }
        
        return block_size;
    }

    void check_a(instruction_ref a) const
    {
        if(a->get_shape().ndim() < 1)
        {
            MIGRAPHX_THROW("MatMulBnb4: Input A must have at least 1 dimension");
        }
    }

    void check_k(instruction_ref a, size_t k) const
    {
        if(a->get_shape().lens().back() != k)
        {
            MIGRAPHX_THROW("MatMulBnb4: Input A inner dimension (" + 
                           std::to_string(a->get_shape().lens().back()) +
                           ") must match attribute K (" + std::to_string(k) + ")");
        }
    }

    void check_b(instruction_ref b, size_t n, size_t k) const
    {
        const size_t expected_b_elements = (n * k + 1) / 2;
        std::vector<size_t> expected_b_lens{expected_b_elements};
        
        if(b->get_shape().lens() != expected_b_lens)
        {
            MIGRAPHX_THROW("MatMulBnb4: Input B does not match expected dims: " +
                           to_string_range(expected_b_lens) +
                           ". Actual dims: " + to_string_range(b->get_shape().lens()));
        }
    }

    void check_absmax(instruction_ref absmax, size_t n, size_t k, size_t block_size) const
    {
        const size_t expected_absmax_elements = (n * k + block_size - 1) / block_size;
        std::vector<size_t> expected_absmax_lens{expected_absmax_elements};
        
        if(absmax->get_shape().lens() != expected_absmax_lens)
        {
            MIGRAPHX_THROW("MatMulBnb4: Input absmax does not match expected dims: " +
                           to_string_range(expected_absmax_lens) +
                           ". Actual dims: " + to_string_range(absmax->get_shape().lens()));
        }
    }

    instruction_ref dequantize_b_bnb4(onnx_parser::node_info& info,
                                      int n,
                                      int k,
                                      int block_size,
                                      int quant_type,
                                      const std::vector<instruction_ref>& args) const
    {
        auto unpacked_b      = unpack_bnb4_data(info, n, k, args[1]);
        auto prepared_absmax = prepare_blockwise_absmax(info, n, k, block_size, args[2]);

        return apply_bnb4_dequantization(info, unpacked_b, prepared_absmax, quant_type);
    }

    instruction_ref
    unpack_bnb4_data(onnx_parser::node_info& info, int n, int k, instruction_ref b) const
    {
        auto unpacked = info.add_instruction(make_op("unpack_int4"), b);
        unpacked      = info.add_instruction(make_op("reshape", {{"dims", {n, k}}}), unpacked);

        return unpacked;
    }

    instruction_ref prepare_blockwise_absmax(
        onnx_parser::node_info& info, int n, int k, int block_size, instruction_ref absmax) const
    {
        auto expanded_absmax = info.add_instruction(make_op("unsqueeze", {{"axes", {1}}}), absmax);

        auto bc_lens    = expanded_absmax->get_shape().lens();
        bc_lens[1]      = block_size;
        expanded_absmax = info.add_instruction(make_op("multibroadcast", {{"out_lens", bc_lens}}),
                                               expanded_absmax);
        expanded_absmax =
            info.add_instruction(make_op("reshape", {{"dims", {n * k}}}), expanded_absmax);
        const size_t total_elements = n * k;
        if(expanded_absmax->get_shape().lens()[0] > total_elements)
        {
            expanded_absmax = info.add_instruction(
                make_op("slice", {{"axes", {0}}, {"starts", {0}}, {"ends", {total_elements}}}),
                expanded_absmax);
        }

        expanded_absmax =
            info.add_instruction(make_op("reshape", {{"dims", {n, k}}}), expanded_absmax);

        return expanded_absmax;
    }

    instruction_ref apply_bnb4_dequantization(onnx_parser::node_info& info,
                                              instruction_ref quantized_data,
                                              instruction_ref absmax,
                                              int quant_type) const
    {
        if(quant_type == 0)
        {
            auto float_data = info.add_instruction(
                make_op("convert", {{"target_type", migraphx::shape::float_type}}), quantized_data);
            auto scale_factor = info.add_literal(
                migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {8.0f}});
            auto scale_factor_bc = info.add_instruction(
                make_op("multibroadcast", {{"out_lens", float_data->get_shape().lens()}}),
                scale_factor);

            auto scaled_data = info.add_instruction(make_op("div"), float_data, scale_factor_bc);
            return info.add_instruction(make_op("mul"), scaled_data, absmax);
        }
        else
        {
            std::vector<float> nf4_lookup_table = {-1.0f,
                                                   -0.6961928009986877f,
                                                   -0.5250730514526367f,
                                                   -0.39491748809814453f,
                                                   -0.28444138169288635f,
                                                   -0.18477343022823334f,
                                                   -0.09105003625154495f,
                                                   0.0f,
                                                   0.07958029955625534f,
                                                   0.16093020141124725f,
                                                   0.24611230194568634f,
                                                   0.33791524171829224f,
                                                   0.44070982933044434f,
                                                   0.5626170039176941f,
                                                   0.7229568362236023f,
                                                   1.0f};
            
            auto lut     = info.add_literal(migraphx::literal{
                migraphx::shape{migraphx::shape::float_type, {16}}, nf4_lookup_table});
            auto indices = info.add_instruction(
                make_op("convert", {{"target_type", migraphx::shape::int64_type}}), quantized_data);
            auto dequant_values =
                info.add_instruction(make_op("gather", {{"axis", 0}}), lut, indices);

            return info.add_instruction(make_op("mul"), dequant_values, absmax);
        }
    }

    instruction_ref matmul(onnx_parser::node_info& info, instruction_ref a, instruction_ref b) const
    {
        const auto a_rank = a->get_shape().ndim();

        if(a_rank == 1)
        {
            a = info.add_instruction(make_op("unsqueeze", {{"axes", {0}}}), a);
        }

        if(a_rank > 2)
        {
            auto b_lens    = b->get_shape().lens();
            auto b_bc_lens = a->get_shape().lens();
            std::copy(b_lens.begin(), b_lens.end(), b_bc_lens.end() - 2);
            b = info.add_instruction(make_op("multibroadcast", {{"out_lens", b_bc_lens}}), b);
        }

        auto dot = info.add_instruction(make_op("dot"), a, b);

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
