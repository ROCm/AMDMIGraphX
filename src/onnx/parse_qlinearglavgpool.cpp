/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/op/pooling.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

/*
 *********************************************************************************
 *  Reference: see QLinearGlobalAveragePool in                                   *
 *  github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md          *
 *********************************************************************************

QLinearGlobalAveragePool consumes an input tensor X and applies
Average pooling across the values in the same channel. This is
equivalent to AveragePool with kernel size equal to the spatial
dimension of input tensor. Input is of type uint8_t or int8_t.

Version
This version of the operator has been available since version 1 of the 'com.microsoft' operator set.

Attributes
channels_last : int

Inputs
X : T

Input data tensor from the previous operator; According to channels_last, dimensions for image case
are (N x C x H x W), or (N x H x W x C) where N is the batch size, C is the number of channels, and
H and W are the height and the width of the data. For non image case, the dimensions are in the form
of (N x C x D1 x D2 ... Dn), or (N x D1 X D2 ... Dn x C) where N is the batch size.

x_scale : tensor(float)
Scale of quantized input 'X'. It must be a scalar.

x_zero_point : T
Zero point tensor for input 'X'. It must be a scalar.

y_scale : tensor(float)
Scale of quantized output 'Y'. It must be a scalar.

y_zero_point : T
Zero point tensor for output 'Y'. It must be a scalar.

Outputs
Y : T
Output data tensor from pooling across the input tensor. The output tensor has the same rank as the
input. with the N and C value keep it value, while the otherdimensions are all 1. Type Constraints
T : tensor(uint8), tensor(int8)
Constrain input and output types to signed/unsigned int8 tensors.

*/

struct parse_qlinearglobalaveragepool : op_parser<parse_qlinearglobalaveragepool>
{
    std::vector<op_desc> operators() const { return {{"QLinearGlobalAveragePool"}}; }

    // basic type checking for QLinearGlobalAveragePool Operator
    void check_inputs(const std::vector<instruction_ref>& args) const
    {
        if(args.size() < 5)
            MIGRAPHX_THROW("QLINEARGLOBALAVERAGEPOOL: missing inputs");

        const auto& in_x      = args[0];
        const auto& zero_pt_x = args[2];
        const auto& zero_pt_y = args[4];

        if(in_x->get_shape().ndim() <= 2)
            MIGRAPHX_THROW("QLINEARGLOBALAVERAGEPOOL: input dimensions too small");

        auto type_x = in_x->get_shape().type();
        if(type_x != migraphx::shape::int8_type and type_x != migraphx::shape::uint8_type)
            MIGRAPHX_THROW("QLINEARGLOBALAVERAGEPOOL: unsupported input type");

        if(type_x != zero_pt_x->get_shape().type())
            MIGRAPHX_THROW("QLINEARGLOBALAVERAGEPOOL: mismatched type: input zero point");

        if(type_x != zero_pt_y->get_shape().type())
            MIGRAPHX_THROW("QLINEARGLOBALAVERAGEPOOL: mismatched type: output zero point");
    }

    // This method is to prep for quantizelinear or dequantizelinear operation for
    // either the broadcasting of weight-scale or zero-points of an operator
    // outputs: operator op (inputs x, broadcasted: scale (float) & zero_pt (8-bit))
    instruction_ref bcast_qdq_instr(const std::string& op_name,
                                    const instruction_ref x_in,
                                    const instruction_ref arg_fscale,
                                    const instruction_ref arg_z_pt,
                                    const onnx_parser::node_info& info) const
    {
        auto in_lens = x_in->get_shape().lens();

        // prep 1: broadcast scale. it can come as a scalar or a 1-D tensor.
        instruction_ref bcast_scale;
        if(arg_fscale->get_shape().elements() > 1)
            bcast_scale = info.add_instruction(
                migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", in_lens}}), arg_fscale);
        else
            bcast_scale = info.add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", in_lens}}), arg_fscale);

        // prep 2: broadcast zero point. it can come as a scalar or a 1-D tensor.
        instruction_ref bcast_zero_pt;
        if(arg_z_pt->get_shape().elements() > 1)
            bcast_zero_pt = info.add_instruction(
                migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", in_lens}}), arg_z_pt);
        else
            bcast_zero_pt = info.add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", in_lens}}), arg_z_pt);

        // op_name is either quantizelinear or dequantizelinear:
        return info.add_instruction(migraphx::make_op(op_name), x_in, bcast_scale, bcast_zero_pt);
    }

    instruction_ref parse(const op_desc& /* opd */,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        check_inputs(args);

        // Input: X
        const auto& in_x      = args[0];
        const auto& scale_x   = args[1];
        const auto& zero_pt_x = args[2];
        auto dquant_x         = bcast_qdq_instr("dequantizelinear", in_x, scale_x, zero_pt_x, info);

        // Output Y = globalaveragepool(X)

        auto op   = migraphx::op::pooling{migraphx::op::pooling_mode::average};
        auto lens = in_x->get_shape().lens();
        std::vector<size_t> lengths(lens.begin() + 2, lens.end());
        op.lengths = lengths;
        op.padding = std::vector<size_t>(lens.size());
        auto out_y = info.add_instruction(op, dquant_x);

        const auto& scale_y   = args[3];
        const auto& zero_pt_y = args[4];

        auto out_quant_y = bcast_qdq_instr("quantizelinear", out_y, scale_y, zero_pt_y, info);

        return out_quant_y;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
