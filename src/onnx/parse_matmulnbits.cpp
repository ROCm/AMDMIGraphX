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
        std::cout << "MatMulNBits parser" << std::endl;
        const auto N = *parse_attribute<int>("N", parser, info);
        const auto K = *parse_attribute<int>("K", parser, info);
        // TODO only 4 is valid
        const auto bits = *parse_attribute<int>("bits", parser, info);
        // TODO check that it is >= 16 and a power of 2
        const auto block_size = *parse_attribute<int>("block_size", parser, info);
        std::cout << "N: " << N << std::endl;
        std::cout << "K: " << K << std::endl;
        std::cout << "bits: " << bits << std::endl;
        std::cout << "block_size: " << block_size << std::endl;

        std::cout << "A shape: " << args[0]->get_shape() << std::endl;
        std::cout << "B shape: " << args[1]->get_shape() << std::endl;

        auto b = info.add_instruction(make_op("reshape", {{"dims", {N, -1}}}), args[1]);
        b      = info.add_instruction(make_op("unpack_int4"), b);
        // Shape: [N x n_blocks_per_col] -> reshape to [N, n_blocks_per_col]
        auto n_blocks_per_col = (K + block_size - 1) / block_size;
        auto scales =
            info.add_instruction(make_op("reshape", {{"dims", {N, n_blocks_per_col}}}), args[2]);
        std::cout << scales->get_shape() << std::endl;
        b = add_dequantize(info, block_size, 1, b, scales);
        std::cout << "b shape after dq: " << b->get_shape() << std::endl;
        b = info.add_instruction(make_op("transpose", {{"permutation", {1, 0}}}), b);
        // Replace with proper matmul
        return info.add_instruction(make_op("dot"), args[0], b);
    }

    private:
    template <typename T>
    std::optional<T> parse_attribute(const std::string& attribute_name,
                                     const onnx_parser& parser,
                                     onnx_parser::node_info& info) const
    {
        if(not contains(info.attributes, attribute_name))
            return std::nullopt;

        return parser.parse_value(info.attributes[attribute_name]).at<T>();
    }

    instruction_ref add_dequantize(onnx_parser::node_info& info,
                                   int block_size,
                                   int axis,
                                   instruction_ref b,
                                   instruction_ref scales) const
    {
        scales = info.add_instruction(make_op("unsqueeze", {{"axes", {axis + 1}}}), scales);

        auto bc_lens      = scales->get_shape().lens();
        bc_lens[axis + 1] = block_size;
        scales = info.add_instruction(make_op("multibroadcast", {{"out_lens", bc_lens}}), scales);
        std::cout << scales->get_shape() << std::endl;

        auto reshape_lens  = b->get_shape().lens();
        reshape_lens[axis] = scales->get_shape().lens()[axis] * block_size;
        scales = info.add_instruction(make_op("reshape", {{"dims", reshape_lens}}), scales);

        // TODO: Runt blocks shouldn't be able to happen, but double check
        // // Detect runt block
        // if(x_lens[axis] < reshape_lens[axis])
        // {
        //     ins = info.add_instruction(
        //         make_op("slice", {{"axes", {axis}}, {"starts", {0}}, {"ends",
        //         {x_lens[axis]}}}), ins);
        // }

        // TODO if zeropoint input is not present, it should not be assumed to be zero, it should be -8
        return info.add_instruction(make_op("dequantizelinear"), {b, scales});
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
