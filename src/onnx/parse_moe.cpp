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
#include <migraphx/onnx/onnx_parser.hpp>
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>

#include <algorithm>
#include <limits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

// Parses the com.microsoft MoE and QMoE (quantized) mixture of experts
// operators into the moe op builder.
struct parse_moe : op_parser<parse_moe>
{
    std::vector<op_desc> operators() const { return {{"MoE"}, {"QMoE"}}; }

    template <class T>
    static T attr(const onnx_parser& parser,
                  const onnx_parser::node_info& info,
                  const std::string& name,
                  T def)
    {
        if(not contains(info.attributes, name))
            return def;
        return parser.parse_value(info.attributes.at(name)).at<T>();
    }

    static std::string
    attr_string(const onnx_parser::node_info& info, const std::string& name, const std::string& def)
    {
        if(not contains(info.attributes, name))
            return def;
        return info.attributes.at(name).s();
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        if(attr<int64_t>(parser, info, "use_sparse_mixer", 0) != 0)
            MIGRAPHX_THROW(opd.onnx_name + ": use_sparse_mixer is not supported");

        value options = {
            {"activation_type", attr_string(info, "activation_type", "relu")},
            {"activation_alpha", attr<float>(parser, info, "activation_alpha", 1.0f)},
            {"activation_beta", attr<float>(parser, info, "activation_beta", 0.0f)},
            {"k", attr<int64_t>(parser, info, "k", 1)},
            {"normalize_routing_weights",
             attr<int64_t>(parser, info, "normalize_routing_weights", 0) != 0},
            {"swiglu_fusion", attr<int64_t>(parser, info, "swiglu_fusion", 0)},
            {"swiglu_limit",
             attr<float>(parser, info, "swiglu_limit", std::numeric_limits<float>::infinity())}};

        std::vector<instruction_ref> builder_args;
        if(opd.onnx_name == "QMoE")
        {
            const auto quant_type = attr_string(info, "quant_type", "int");
            if(quant_type != "int")
                MIGRAPHX_THROW("QMoE: quant_type " + quant_type +
                               " is not supported, only integer quantization");
            const auto bits = attr<int64_t>(parser, info, "expert_weight_bits", 4);
            if(bits != 4 and bits != 8)
                MIGRAPHX_THROW("QMoE: expert_weight_bits must be 4 or 8, got " +
                               std::to_string(bits));
            options["expert_weight_bits"] = bits;

            if(args.size() < 6)
                MIGRAPHX_THROW("QMoE: expected at least 6 inputs");
            // The QMoE input order matches the builder's argument layout
            constexpr std::size_t builder_slots = 14;
            const auto n                        = std::min(args.size(), builder_slots);
            // Inputs beyond the fc weights/scales/biases/zero points:
            // router_weights and the fp4/fp8 global and activation scales
            if(std::any_of(args.begin() + n, args.end(), [](instruction_ref a) {
                   return not a->is_undefined();
               }))
                MIGRAPHX_THROW("QMoE: router_weights and global/activation scale inputs "
                               "are not supported");
            args.resize(n);
            builder_args = std::move(args);
        }
        else
        {
            if(args.size() < 5 or args.size() > 8)
                MIGRAPHX_THROW("MoE: expected between 5 and 8 inputs");
            // Remap the MoE input order {input, router_probs, fc1_weights, fc1_bias,
            // fc2_weights, fc2_bias, fc3_weights, fc3_bias} into the builder layout
            // with undefined scale slots
            auto undef = info.add_instruction(make_op("undefined"));
            args.resize(8, undef);
            builder_args = {args[0],
                            args[1],
                            args[2],
                            undef,
                            args[3],
                            args[4],
                            undef,
                            args[5],
                            args[6],
                            undef,
                            args[7]};
        }
        return op::builder::add("moe", *info.mod, builder_args, options).at(0);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
