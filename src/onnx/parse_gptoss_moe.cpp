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
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

// Parses the contrib node "GptOssMoE" emitted by the model rewriter into a
// single migraphx gptoss_moe op. The rewriter has already repacked the expert
// weights into the kernel-native layout:
//   in0 hidden_states  (io_dtype, [S, hidden])
//   in1 router_logits  (f32, [S, num_experts])
//   in2 fc1_weights    (uint32, [E, 2*inter, hidden/8])
//   in3 fc1_scales     (f32,    [E, 2*inter, hidden/32])
//   in4 fc2_weights    (uint32, [E, hidden, inter/8])
//   in5 fc2_scales     (f32,    [E, hidden, inter/32])
//
// Activations run FP32 inside the op (the kernels are FP32); we insert
// FP16->FP32 in and FP32->FP16 out so the surrounding graph stays io_dtype.
struct parse_gptoss_moe : op_parser<parse_gptoss_moe>
{
    std::vector<op_desc> operators() const { return {{"GptOssMoE"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto get_int = [&](const std::string& k, int def) {
            return contains(info.attributes, k)
                       ? parser.parse_value(info.attributes.at(k)).at<int>()
                       : def;
        };
        auto get_float = [&](const std::string& k, float def) {
            return contains(info.attributes, k)
                       ? parser.parse_value(info.attributes.at(k)).at<float>()
                       : def;
        };

        const int num_experts       = get_int("num_experts", 32);
        const int top_k             = get_int("top_k", 4);
        const int hidden_size       = get_int("hidden_size", 2880);
        const int intermediate_size = get_int("intermediate_size", 2880);
        const float swiglu_alpha    = get_float("swiglu_alpha", 1.702f);
        const float swiglu_beta     = get_float("swiglu_beta", 1.0f);
        const float swiglu_limit    = get_float("swiglu_limit", 7.0f);

        auto io_type = args.at(0)->get_shape().type();

        // hidden_states FP16 -> FP32 for the kernels
        if(io_type != shape::float_type)
        {
            args[0] = info.add_instruction(
                make_op("convert", {{"target_type", shape::float_type}}), args[0]);
        }

        auto moe = info.add_instruction(make_op("gptoss_moe",
                                                {{"num_experts", num_experts},
                                                 {"top_k", top_k},
                                                 {"hidden_size", hidden_size},
                                                 {"intermediate_size", intermediate_size},
                                                 {"swiglu_alpha", swiglu_alpha},
                                                 {"swiglu_beta", swiglu_beta},
                                                 {"swiglu_limit", swiglu_limit}}),
                                        args);

        // Convert fp32 op output -> io_dtype (fp16), matching the proven concat-dense
        // path (both SkipLayerNorm inputs fp16, as ORT's schema requires).
        if(io_type != shape::float_type)
        {
            moe = info.add_instruction(make_op("convert", {{"target_type", io_type}}), moe);
        }
        return moe;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
