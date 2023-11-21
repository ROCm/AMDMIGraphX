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
#include <migraphx/onnx/padding.hpp>
#include <migraphx/onnx/conv.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/onnx/broadcast_qdq.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

/*
 *********************************************************************************
 *  Reference: see QLinearConcat in                                              *
 *  https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md  *
 *********************************************************************************

com.microsoft.QLinearConcat
Concatenate a list of tensors into a single tensor.All input tensors must have the same shape,
except for the dimension size of the axis to concatenate on.

Version
This version of the operator has been available since version 1 of the 'com.microsoft' operator set.

Attributes
axis : int (required)
Which axis to concat on

Inputs (3 - ∞)
Y_scale : TF
Y's scale.

Y_zero_point : T8
Y's zero point.

inputs (variadic, heterogeneous) : TV
List of tensors/scale/zero_point for concatenation

Outputs
Y : T8
Concatenated tensor

Type Constraints
T8 : tensor(uint8), tensor(int8)
Constrain input and output types to 8 bit signed and unsigned tensors.

TF : tensor(float)
Constrain scale types to any float tensor type.

TV : tensor(uint8), tensor(int8), tensor(float)
Sequence of (Tensor, Scale, ZeroPoint) tuples. The type is sequence of (T8, TF, T8).
*/

struct parse_qlinearconcat : op_parser<parse_qlinearconcat>
{
    std::vector<op_desc> operators() const { return {{"QLinearConcat"}}; }

    // basic type checking for QLinearConcat Operator
    void check_inputs(const std::vector<instruction_ref>& args) const
    {
        auto args_size = args.size();
        // at least 5 input tensors, can be 5, 8, 11 ...
        if((args_size < 5) or ((args_size - 2) % 3 != 0))
            MIGRAPHX_THROW("QLINEARCONCAT: missing inputs");

        auto y_zp      = args[1];
        auto y_zp_type = y_zp->get_shape().type();
        if(y_zp_type != migraphx::shape::int8_type and y_zp_type != migraphx::shape::uint8_type)
            MIGRAPHX_THROW("QLINEARCONCAT: unsupported output type");

        auto t0_type = args[2]->get_shape().type();
        if(t0_type != migraphx::shape::int8_type and t0_type != migraphx::shape::uint8_type)
            MIGRAPHX_THROW("QLINEARCONCAT: unsupported input type");
        for(auto idx = 2; idx < args.size(); idx += 3)
        {
            if((args[idx]->get_shape().type() != t0_type) or
               (args[idx + 2]->get_shape().type() != t0_type))
            {
                MIGRAPHX_THROW("QLINEARCONCAT: mismatching input types");
            }
        }
    }

    instruction_ref parse(const op_desc& /* opd */,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        check_inputs(args);
        if(not contains(info.attributes, "axis"))
            MIGRAPHX_THROW("QLINEARCONCAT: missing axis attribute");

        auto axis = parser.parse_value(info.attributes.at("axis")).template at<int64_t>();
        std::vector<instruction_ref> tmp;
        for(auto idx = 2; idx < args.size(); idx += 3)
        {
            auto data_tensor = args[idx];
            auto scale       = args[idx + 1];
            auto zero_pt     = args[idx + 2];
            tmp.push_back(bcast_qdq_instr("dequantizelinear", data_tensor, scale, zero_pt, info));
        }
        auto y = info.add_instruction(migraphx::make_op("concat", {{"axis", axis}}), tmp);

        auto y_scale   = args[0];
        auto y_zero_pt = args[1];

        return bcast_qdq_instr("quantizelinear", y, y_scale, y_zero_pt, info);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
