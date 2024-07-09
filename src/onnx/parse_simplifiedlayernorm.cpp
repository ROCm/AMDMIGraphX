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
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

// Ported CPU implementation from 
// https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/nn/layer_norm_impl.cc

struct parse_simplifiedlayernorm : op_parser<parse_simplifiedlayernorm>
{
    std::vector<op_desc> operators() const { return {{"SimplifiedLayerNormalization"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       const onnx_parser::node_info& info,
                                       std::vector<instruction_ref> args) const
    {
        int64_t axis = -1;
        if(contains(info.attributes, "axis"))
        {
            axis = parser.parse_value(info.attributes.at("axis")).at<int64_t>();
        }
        float epsilon = 1e-5f;
        if(contains(info.attributes, "epsilon"))
        {
            epsilon = parser.parse_value(info.attributes.at("epsilon")).at<float>();
        }
        if(contains(info.attributes, "stash_type"))
        {
            std::cerr << "WARNING: LAYERNORM does not support stash_type, it will be ignored.\n";
        }

        if(args.size() != 2)
        {
            MIGRAPHX_THROW("PARSE_LAYERNORM: invalid input count");
        }

        auto x        = args.at(0);
        auto scale     = args.at(1);

        auto x_shape   = x->get_shape();
        auto x_dtype   = x_shape.type();
        int64_t x_rank = x_shape.ndim();
        axis = axis < 0 ? axis + x_rank : axis;

        if(x_rank < 2 or x_rank > 3)
        {
            MIGRAPHX_THROW("PARSE_LAYERNORM: invalid input shape");
        }
        
        auto x_sq = info.add_common_op("mul", x, x);
        auto rms = info.add_instruction(make_op("reduce_mean", {{"axes", {axis}}}), x_sq);
        auto mean = rms;
        epsilon =
            (x_dtype == migraphx::shape::half_type and std::abs(epsilon) < 1e-7) ? 1e-7 : epsilon;
        auto eps     = info.add_literal(migraphx::literal{migraphx::shape{x_dtype}, {epsilon}});
        rms = info.add_common_op("add", rms, eps);
        auto rrms = info.add_instruction(make_op("rsqrt"), rms);
        auto result = info.add_common_op("mul", x, rrms);
        result = info.add_common_op("mul", result, scale);

        return {result, mean, rrms};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
