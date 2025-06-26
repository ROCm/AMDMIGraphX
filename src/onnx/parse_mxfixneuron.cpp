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
        if(args.size() != 1)
        {
            MIGRAPHX_THROW("MXFixNeuron: must have only 1 input");
        }
        int block_axis = info.attributes.at("axis").i();
        if(block_axis == 0)
        {
            MIGRAPHX_THROW("MXFixNeuron: block dimension should not be 0 dimension. Expecting 0 "
                           "dimension to be batch dimension.");
        }
        int block_size = info.attributes.at("block_size").i();
        if(block_size != 32)
        {
            MIGRAPHX_THROW("MXFixNeuron: only block_size of 32 is supported");
        }
        std::string element_dtype = info.attributes.at("element_dtype").s();
        if(element_dtype != "fp4_e2m1")
        {
            MIGRAPHX_THROW("MXFixNeuron: only support MXFP4 type");
        }
        int rounding_mode = info.attributes.at("rounding_mode").i();
        if(rounding_mode != 2)
        {
            MIGRAPHX_THROW("MXFixNeuron: only round ties to even is supported");
        }

        // make reduction axes for calculating block scales
        const auto input_shape = args.front()->get_shape();
        const auto input_lens  = input_shape.lens();
        if(input_lens.at(block_axis) % block_size != 0)
        {
            MIGRAPHX_THROW(
                "MXFixNeuron: only support block axis being evenly divisible by block_size");
        }
        auto scale_dim                       = input_lens.at(block_axis) / block_size;
        std::vector<std::size_t> reduct_dims = input_lens;
        reduct_dims.at(block_axis) /= scale_dim;
        reduct_dims.insert(reduct_dims.begin() + block_axis, scale_dim);
        instruction_ref reshape_ins =
            info.add_instruction(make_op("reshape", {{"dims", reduct_dims}}), args.front());
        // reduce over all axes not batch or block dimension
        std::vector<std::int64_t> reduct_axes;
        for(auto i = 1; i < input_shape.ndim(); ++i)
        {
            if(i != block_axis)
                reduct_axes.push_back(i);
        }

        // dynamic quantization:
        // V_k = fp32 vector input of block size k
        // B_k = pow(2, floor(log2(reduce_max(V_k)))) # largest power of 2 less than V
        // X_k = block scale k = B_k / (largest power of 2 in fp4e2m1) = B_k / 4
        auto reduce_max_ins =
            info.add_instruction(make_op("reduce_max", {{"axes", reduct_axes}}), reshape_ins);
        auto abs_ins   = info.add_instruction(make_op("abs"), reduce_max_ins);
        auto log2_ins  = info.add_instruction(make_op("log2"), abs_ins);
        auto floor_ins = info.add_instruction(make_op("floor"), log2_ins);
        auto exp_ins   = info.add_instruction(make_op("exp"), floor_ins);
        auto lit_4_ins = info.add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {4.f}});
        auto broadcast_lit_ins = info.add_instruction(
            make_op("multibroadcast", {{"dims", reduce_max_ins->get_shape().lens()}}), lit_4_ins);
        auto block_scales_ins = info.add_instruction(make_op("div"), exp_ins, broadcast_lit_ins);
        // broadcast scales for use in quantizelinear
        auto broadcast_scales_ins = info.add_instruction(
            make_op("multibroadcast", {{"dims", reduct_dims}}), block_scales_ins);
        auto reshape_scales_ins =
            info.add_instruction(make_op("reshape", {{"dims", input_lens}}), broadcast_scales_ins);
        auto q_ins =
            info.add_instruction(make_op("quantizelinear"), args.front(), reshape_scales_ins);
        auto dq_ins = info.add_instruction(make_op("dequantizelinear"), q_ins, reshape_scales_ins);
        // NOTE: scales and q-dq all still in float_type
        return dq_ins;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
