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
#include <migraphx/onnx/matmulnbits.hpp>
#include <migraphx/ranges.hpp>
#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

// Parses MatMulNBits from both com.microsoft and com.ryzenai.
// Ryzen AI hybrid graphs may append optional bias and an NPU-only packed weight
// input after the microsoft schema (A, B/qweight, scales, zero_points).
struct parse_matmulnbits : op_parser<parse_matmulnbits>
{
    std::vector<op_desc> operators() const { return {{"MatMulNBits"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        if(args.size() < 3)
            MIGRAPHX_THROW("MatMulNBits: expects at least 3 inputs (A, B, scales), got " +
                           std::to_string(args.size()));

        const int n          = parse_attribute(parser, info, "N");
        const int k          = parse_attribute(parser, info, "K");
        const int bits       = parse_attribute(parser, info, "bits");
        const int block_size = parse_attribute(parser, info, "block_size");

        std::optional<instruction_ref> zp;
        if(args.size() > 3 and not args[3]->is_undefined())
            zp = args[3];

        std::optional<instruction_ref> bias;
        if(args.size() > 4 and not args[4]->is_undefined())
            bias = args[4];

        // args[5] (qweight.packed), when present, is NPU-only and ignored here.
        return matmulnbits::add_matmul_nbits(
            info, args[0], args[1], args[2], zp, bias, n, k, bits, block_size);
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
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
