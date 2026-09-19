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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

// com.microsoft.SkipSimplifiedLayerNormalization
// Skip and Root Mean Square Layer Normalization

// Version
// This version of the operator has been available since version 1 of the 'com.microsoft' operator
// set.

// Type Constraints
// T : tensor(float), tensor(float16)
// Constrain input and output types to float or half tensors.
// U : tensor(float)
// Constrain mean and inv_std_var to float tensors.

struct parse_skip_simplified_layer_normalization
    : op_parser<parse_skip_simplified_layer_normalization>
{
    std::vector<op_desc> operators() const { return {{"SkipSimplifiedLayerNormalization"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       std::vector<instruction_ref> args) const
    {
        // Attributes
        // epsilon : float
        // The epsilon value to use to avoid division by zero.
        float epsilon = 1e-5f;
        if(contains(info.attributes, "epsilon"))
        {
            epsilon = parser.parse_value(info.attributes.at("epsilon")).at<float>();
        }

        // Inputs (3 - 4)
        // input : T
        // 3D input tensor with shape (batch_size, sequence_length, hidden_size) Or 2D input tensor
        // with shape (token_count, hidden_size)
        // skip : T
        // 3D input tensor with shape (batch_size, sequence_length, hidden_size)
        // Or 2D input tensor with shape (token_count, hidden_size)
        // gamma : T
        // 1D input tensor with shape (hidden_size)
        // bias (optional) : T
        // 1D bias tensor with shape (hidden_size) - not used by ORT

        if(args.size() < 3 or args.size() > 4)
        {
            MIGRAPHX_THROW("PARSE_SKIPSIMPLIFIEDLAYERNORMALIZATION: invalid input count");
        }

        auto x     = args.at(0);
        auto skip  = args.at(1);
        auto gamma = args.at(2);
        instruction_ref bias;
        if(args.size() == 4)
        {
            bias = args.at(3);
        }

        auto x_shape       = x->get_shape();
        auto x_dtype       = x_shape.type();
        int64_t x_rank     = x_shape.ndim();
        int64_t skip_rank  = skip->get_shape().ndim();
        int64_t gamma_rank = gamma->get_shape().ndim();
        // axis = hidden_size dim
        int64_t axis = x_rank - 1;

        if(x_rank < 2 or x_rank > 3 or x_rank != skip_rank or gamma_rank != 1)
        {
            MIGRAPHX_THROW("PARSE_SKIPSIMPLIFIEDLAYERNORMALIZATION: invalid input shape");
        }

        x         = info.add_common_op("add", x, skip);
        // Convert to float before reduce_mean
        // Fp16 reduce_mean on GPU causes loss of accuracy
        auto float_x = info.add_instruction(
            make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto x_sq = info.add_common_op("mul", float_x, float_x);
        auto rms  = info.add_instruction(make_op("reduce_mean", {{"axes", {axis}}}), x_sq);
        // Full FP32 normalization: mirror the DML reference (ComputeSkipSLNCPU in
        // dml/hip_qmoe/qmoe_hip_combined_op.cpp): mean_sq in FP32, inv_std in FP32,
        // x*inv_std*gamma all in FP32, only the final output cast back to io_dtype.
        // Converting rrms or intermediate results to FP16 early reintroduces the
        // precision loss that causes router logits to drift 3-10x by layer 20.
        auto mean = info.add_instruction(
            make_op("convert", {{"target_type", x_dtype}}), rms); // FP16 mean for output only
        epsilon =
            (x_dtype == migraphx::shape::half_type and std::abs(epsilon) < 1e-7) ? 1e-7 : epsilon;
        auto eps_f32  = info.add_literal(migraphx::literal{migraphx::shape{migraphx::shape::float_type}, {epsilon}});
        auto rms_ep   = info.add_common_op("add", rms, eps_f32);         // FP32
        auto rrms_f32 = info.add_instruction(make_op("rsqrt"), rms_ep);  // FP32
        // Cast gamma to FP32 so mul stays in FP32
        // Use contiguous to anchor the FP32 value and prevent eliminate_convert from
        // removing the cast when gamma is used elsewhere in FP16.
        auto gamma_f32 = info.add_instruction(
            make_op("convert", {{"target_type", migraphx::shape::float_type}}), gamma);
        gamma_f32 = info.add_instruction(make_op("contiguous"), gamma_f32);
        // Compute x*rrms*gamma entirely in FP32 (add_common_op handles broadcasting)
        auto result_f32 = info.add_common_op("mul", float_x, rrms_f32);
        result_f32      = info.add_common_op("mul", result_f32, gamma_f32);
        // Cast final result back to io_dtype
        auto rrms = info.add_instruction(
            make_op("convert", {{"target_type", x_dtype}}), rrms_f32); // kept for output slot
        auto result = info.add_instruction(
            make_op("convert", {{"target_type", x_dtype}}), result_f32);
        if(args.size() == 4)
        {
            result = info.add_common_op("add", result, bias);
            x      = info.add_common_op("add", x, bias);
        }

        // Outputs (1 - 4)
        // output : T
        // 3D output tensor with shape (batch_size, sequence_length, hidden_size)Or 2D output tensor
        // with shape (token_count, hidden_size)
        // mean (optional) : U Saved mean used during training
        // to speed up gradient computation
        // inv_std_var (optional) : U Saved inverse standard
        // variance used during training to speed up gradient computation.
        // input_skip_bias_sum (optional) : T Sum of the input and skip inputs (and bias if it
        // exists)with shape (batch_size, sequence_length, hidden_size) or (token_count,
        // hidden_size).

        return {result, mean, rrms, x};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
