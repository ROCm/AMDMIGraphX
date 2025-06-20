/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_mxfixneuron : op_parser<parse_mxfixneuron>
{
    std::vector<op_desc> operators() const { return {{"MXFixNeuron"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
		/**
		 * TODO
		 * Parse the ONNX op into a fake quantized Q/DQ pair with fp4 type as output of quantizelinear.
		 * Also have to do the dynamic quantization steps.
		 * simplify_qdq pass will add pack and unpack. Then it will then do the real quantization by
		 * moving the DQ and recalculating scales and zero_points.
		 *
		 * dynamic quantization:
		 * V_k = fp32 vector input of block size k
		 * B_k = pow(2, floor(log2(reduce_max(V_k)))) # largest power of 2 less than V
		 * probably can also do: B = most significant exponent of max(V)
		 * but don't have the bit operators to do this (bit_or and bitshifts)
		 * X_k = block scale k = B_k / (largest power of 2 in fp4e2m1) = B_k / 4
		 * 
		 * Reshape and broadcast the scales for quantization operators to be linear.
		 * Quantize and dequantize with the reshaped and broadcasted scales.
		 */
		


		
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
