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
#include "migraphx/functional.hpp"
#include "migraphx/instruction_ref.hpp"
#include "migraphx/onnx/onnx_parser.hpp"
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_matmulnbits : op_parser<parse_matmulnbits>
{
    std::vector<op_desc> operators() const { return {{"MatMulNBits"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        const int N          = parse_attribute(parser, info, "N");
        const int K          = parse_attribute(parser, info, "K");
        const int bits       = parse_attribute(parser, info, "bits");
        const int block_size = parse_attribute(parser, info, "block_size");

        if(bits != 4)
            MIGRAPHX_THROW("MatMulNBits: bits only supported for value of 4, actual value " +
                           std::to_string(bits));

        if(block_size < 16 and block_size % 2 != 0)
            MIGRAPHX_THROW("MatMulNBits: block_size must be a power of 2 and greater or equal to "
                           "16, actual value " +
                           std::to_string(block_size));

        int n_blocks_per_col = (K + block_size - 1) / block_size;
        int blob_size        = std::ceil(block_size * bits / 8.0f);

        std::vector<size_t> expected_b_lens{static_cast<size_t>(N),
                                            static_cast<size_t>(n_blocks_per_col),
                                            static_cast<size_t>(blob_size)};
        if(args[1]->get_shape().lens() != expected_b_lens)
            MIGRAPHX_THROW("Input B does not match expected dims TODO");

        std::vector<size_t> expected_scales_lens{static_cast<size_t>(N * n_blocks_per_col)};
        if(args[2]->get_shape().lens() != expected_scales_lens)
            MIGRAPHX_THROW("Input Scales does not match expected dims TODO");

        if(args.size() > 3)
        {
            std::vector<size_t> expected_zp_lens{
                static_cast<size_t>(N * std::ceil(n_blocks_per_col * bits / 8.0f))};
            if(args[3]->get_shape().lens() != expected_zp_lens)
                MIGRAPHX_THROW("MatMulNBits: TODO");
        }

        auto b = dequantize_b(info, N, K, block_size, args);

        b = info.add_instruction(make_op("transpose", {{"permutation", {1, 0}}}), b);
        // Replace with proper matmul
        return info.add_instruction(make_op("dot"), args[0], b);
    }

    private:
    int parse_attribute(const onnx_parser& parser,
                        onnx_parser::node_info& info,
                        const std::string& attribute_name) const
    {
        if(not contains(info.attributes, attribute_name))
            MIGRAPHX_THROW("MatMulNBits: Attribute " + attribute_name +
                           " required, but is missing");

        return parser.parse_value(info.attributes[attribute_name]).at<int>();
    }

    instruction_ref dequantize_b(onnx_parser::node_info& info,
                                 int N,
                                 int K,
                                 int block_size,
                                 const std::vector<instruction_ref>& args) const
    {
        auto b = unpack(info, args[1], N, K);

        auto n_blocks_per_col = (K + block_size - 1) / block_size;
        auto scales = info.add_instruction(make_op("reshape", {{"dims", {N, -1}}}), args[2]);
        scales      = prepare_blockwise_dq_arg(info, scales, N, K, block_size);

        instruction_ref zp;
        if(args.size() == 4)
        {
            zp = unpack(info, args[3], N, n_blocks_per_col);
            zp = prepare_blockwise_dq_arg(info, zp, N, K, block_size);
        }
        else
        {
            zp = info.add_literal(literal{shape{shape::uint8_type, {1}}, {8}});
            zp = info.add_instruction(
                make_op("multibroadcast", {{"out_lens", b->get_shape().lens()}}), zp);
        }
        return info.add_instruction(make_op("dequantizelinear"), {b, scales, zp});
    }

    instruction_ref unpack(onnx_parser::node_info& info, instruction_ref x, int N, int dim1) const
    {
        x = info.add_instruction(make_op("unpack_int4"), x);
        x = info.add_instruction(make_op("reshape", {{"dims", {N, -1}}}), x);
        if(x->get_shape().lens()[1] > dim1)
        {
            x = info.add_instruction(
                make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {dim1}}}), x);
        }
        return x;
    }

    instruction_ref prepare_blockwise_dq_arg(
        onnx_parser::node_info& info, instruction_ref x, int N, int K, int block_size) const
    {
        x = info.add_instruction(make_op("unsqueeze", {{"axes", {2}}}), x);

        auto bc_lens = x->get_shape().lens();
        bc_lens[2]   = block_size;
        x            = info.add_instruction(make_op("multibroadcast", {{"out_lens", bc_lens}}), x);
        x            = info.add_instruction(make_op("reshape", {{"dims", {N, -1}}}), x);

        // Detect runt block
        if(x->get_shape().lens()[1] > K)
        {
            x = info.add_instruction(
                make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {K}}}), x);
        }

        return x;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
