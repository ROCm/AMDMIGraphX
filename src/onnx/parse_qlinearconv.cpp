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
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

/*
 *********************************************************************************
 *  Reference: see QLinearConv in                                                *
 *  https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md  *
 *********************************************************************************

com.microsoft.QLinearConv

Version
This version of the operator has been available since version 1 of the 'com.microsoft' operator set.

Attributes

auto_pad : string

channels_last : int

dilations : list of ints

group : int

kernel_shape : list of ints

pads : list of ints

strides : list of ints

Inputs (8 - 9)

x : T1

x_scale : tensor(float)

x_zero_point : T1

w : T2

w_scale : tensor(float)

w_zero_point : T2

y_scale : tensor(float)

y_zero_point : T3

B (optional) : T4

Outputs
y : T3

Type Constraints
T1 : tensor(int8), tensor(uint8)
T2 : tensor(int8), tensor(uint8)
T3 : tensor(int8), tensor(uint8)
T4 : tensor(int32)

More details also at:
https://xadupre.github.io/draft/onnx/onnx_doc_folder/onnx__QLinearConv.html

*/

struct parse_qlinearconv : op_parser<parse_qlinearconv>
{
    std::vector<op_desc> operators() const { return {{"QLinearConv"}}; }

    // basic type checking for QLinearConv Operator
    void check_inputs(const std::vector<instruction_ref>& inp_arg) const
    {
        if(inp_arg.size() < 8)
            MIGRAPHX_THROW("QLINEARCONV: missing inputs");

        const instruction_ref& in_x       = inp_arg[0];
        const instruction_ref& in_scale_x = inp_arg[1];
        const instruction_ref& in_w       = inp_arg[3];
        const instruction_ref& in_scale_w = inp_arg[4];
        const instruction_ref& in_scale_y = inp_arg[6];

        auto sh_x   = in_x->get_shape();
        auto sh_w   = in_w->get_shape();
        auto type_x = sh_x.type();
        auto type_w = sh_w.type();

        if(type_x != shape::int8_type and type_x != shape::uint8_type)
            MIGRAPHX_THROW("QLINEARCONV: unsupported input type");
        if(type_w != shape::int8_type and type_w != shape::uint8_type)
            MIGRAPHX_THROW("QLINEARCONV: unsupported weight type");
        if(in_scale_x->get_shape().type() != shape::float_type)
            MIGRAPHX_THROW("QLINEARCONV x scale type should be float");
        if(in_scale_w->get_shape().type() != shape::float_type)
            MIGRAPHX_THROW("QLINEARCONV: wt scale type should be float");
        if(in_scale_y->get_shape().type() != shape::float_type)
            MIGRAPHX_THROW("QLINEARCONV: y scale type should be float");
        if(inp_arg.size() > 8 and inp_arg[8]->get_shape().type() != shape::int32_type)
            MIGRAPHX_THROW("QLINEARCONV y bias should be int32");
    }

    // process all attributes of QLinearConv Operator..
    value process_attributes(const onnx_parser& parser,
                             const onnx_parser::node_info& info,
                             const std::vector<instruction_ref>& args) const
    {
        value values;
        const auto& in_x = args[0];
        auto in_x_shape  = in_x->get_shape();
        auto in_x_lens   = in_x_shape.max_lens();
        assert(in_x_lens.size() > 2);
        auto kdims = in_x_lens.size() - 2;

        check_padding_mode(info, "QLINEARCONV");

        if(contains(info.attributes, "strides"))
        {
            copy(info.attributes.at("strides").ints(), std::back_inserter(values["stride"]));
            check_attr_sizes(kdims, values["stride"].size(), "QLINEARCONV: inconsistent strides");
        }

        if(contains(info.attributes, "dilations"))
        {
            copy(info.attributes.at("dilations").ints(), std::back_inserter(values["dilation"]));
            check_attr_sizes(
                kdims, values["dilation"].size(), "QLINEARCONV: inconsistent dilations");
        }

        if(contains(info.attributes, "pads"))
        {
            std::vector<int64_t> padding;
            copy(info.attributes.at("pads").ints(), std::back_inserter(padding));
            check_attr_sizes(kdims, padding.size() / 2, "QLINEARCONV: inconsistent padding");
            values["padding"] = std::vector<size_t>(padding.begin(), padding.end());
        }

        if(contains(info.attributes, "auto_pad"))
        {
            auto auto_pad = info.attributes.at("auto_pad").s();
            if(auto_pad.find("NOTSET") == std::string::npos)
                MIGRAPHX_THROW("QLINEARCONV: auto_pad option fix..");
        }

        if(contains(info.attributes, "group"))
            values["group"] = parser.parse_value(info.attributes.at("group")).template at<int>();

        if(not values.empty())
            recalc_conv_attributes(values, kdims);

        return values;
    }

    // This method is to prep for quantizelinear or dequantizelinear operation for
    // either the broadcasting of weight-scale or zero-points of qlinearconv operator
    // outputs: operator op (inputs x, broadcasted: scale (float) & zero_pt (8-bit))
    instruction_ref broadcast_instr(const std::string& op_name,
                                    const instruction_ref& x_in,
                                    const instruction_ref& arg_fscale,
                                    const instruction_ref& arg_z_pt,
                                    const onnx_parser::node_info& info) const
    {
        auto in_lens = x_in->get_shape().lens();

        // prep 1: broadcast scale. it can come as a scalar or a 1-D tensor.
        std::vector<float> sc_val;
        auto ev_arg_fscale = arg_fscale->eval();
        ev_arg_fscale.visit([&](auto s) { sc_val.assign(s.begin(), s.end()); });
        shape sh_scale = {shape::float_type, {sc_val.size()}};
        instruction_ref bcast_scale;
        if(sc_val.size() > 1)
            bcast_scale = info.add_instruction(
                migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", in_lens}}),
                info.add_literal(sh_scale, sc_val));
        else
            bcast_scale =
                info.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", in_lens}}),
                                     info.add_literal(sh_scale, sc_val));

        // prep 2: broadcast zero point. it can come as a scalar or a 1-D tensor.
        std::vector<int> z_pt_val;
        auto ev_arg_z_pt = arg_z_pt->eval();
        ev_arg_z_pt.visit([&](auto z) { z_pt_val.assign(z.begin(), z.end()); });
        instruction_ref bcast_zero_pt;
        if(z_pt_val.size() > 1)
            bcast_zero_pt = info.add_instruction(
                migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", in_lens}}), arg_z_pt);
        else
            bcast_zero_pt = info.add_instruction(
                migraphx::make_op("multibroadcast", {{"out_lens", in_lens}}), arg_z_pt);

        // op_name is either quantizelinear or dequantizelinear:
        return info.add_instruction(migraphx::make_op(op_name), x_in, bcast_scale, bcast_zero_pt);
    }

    instruction_ref add_bias_to_conv(const instruction_ref& bias_arg,
                                     const instruction_ref& conv_instr,
                                     const onnx_parser::node_info& info) const
    {
        auto conv_sh   = conv_instr->get_shape();
        auto conv_lens = conv_sh.lens();
        auto conv_type = conv_sh.type();

        auto broadcast_bias = info.add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", conv_lens}}), bias_arg);
        auto f_bias =
            info.add_instruction(make_op("convert", {{"target_type", conv_type}}), broadcast_bias);

        return info.add_instruction(migraphx::make_op("add"), conv_instr, f_bias);
    };

    instruction_ref parse(const op_desc& /* opd */,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        check_inputs(args);

        auto values = process_attributes(parser, info, args);

        // input: quantized x, scale, zero_pt
        const instruction_ref& in_x         = args[0];
        const instruction_ref& in_scale_x   = args[1];
        const instruction_ref& in_zero_pt_x = args[2];

        // input: quantized weights, scale, zero_pt
        const instruction_ref& in_w         = args[3];
        const instruction_ref& in_scale_w   = args[4];
        const instruction_ref& in_zero_pt_w = args[5];

        // for the dequantized output  y: scale & zero_pt
        const instruction_ref& in_scale_y   = args[6];
        const instruction_ref& in_zero_pt_y = args[7];

        auto dquant_x = broadcast_instr("dequantizelinear", in_x, in_scale_x, in_zero_pt_x, info);

        auto dquant_w = broadcast_instr("dequantizelinear", in_w, in_scale_w, in_zero_pt_w, info);

        auto conv_op = values.empty() ? migraphx::make_op("convolution")
                                      : migraphx::make_op("convolution", values);

        auto conv_x_w = info.add_instruction(conv_op, dquant_x, dquant_w);

        // Biases, if any.. : is an optional argument.
        if(args.size() > 8)
            conv_x_w = add_bias_to_conv(args[8], conv_x_w, info);

        auto quant_conv =
            broadcast_instr("quantizelinear", conv_x_w, in_scale_y, in_zero_pt_y, info);
        return quant_conv;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
