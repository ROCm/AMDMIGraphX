/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/onnx/checks.hpp>
#include <migraphx/onnx/conv.hpp>
#include <migraphx/onnx/padding.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_convolution : op_parser<parse_convolution>
{
    std::vector<op_desc> operators() const
    {
        return {{"Conv", "convolution"},
                {"ConvInteger", "quant_convolution"},
                {"NhwcConv", "convolution"}};
    }

    // Convert to half prior to a shift to ensure we preserve accuracy here then
    // convert back to int8
    static instruction_ref add_int8_shift(const onnx_parser::node_info& info,
                                          const instruction_ref& offset_op,
                                          instruction_ref& unshifted_input)
    {
        auto unshifted_input_half = info.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
            unshifted_input);

        auto input_shifted_half = info.add_common_op("add", unshifted_input_half, offset_op);

        return info.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::int8_type}}),
            input_shifted_half);
    }

    static void shift_input_and_bias(const onnx_parser::node_info& info,
                                     const instruction_ref& offset_op,
                                     const bool has_bias,
                                     instruction_ref& input,
                                     instruction_ref& input_bias)
    {
        input = add_int8_shift(info, offset_op, input);
        if(has_bias)
        {
            input_bias = add_int8_shift(info, offset_op, input_bias);
        }
    }

    static float get_symmetric_value(const instruction_ref& input)
    {
        float symmetric_value = 0;
        // adjust symmetric zero point value for uint8 types
        if(input->get_shape().type() == migraphx::shape::uint8_type)
        {
            symmetric_value = 128;
        }
        return symmetric_value;
    }

    static instruction_ref gen_symmetric_literal(const instruction_ref& input,
                                                 const bool is_quant_conv,
                                                 onnx_parser::node_info& info)
    {
        instruction_ref ret = input;
        if(is_quant_conv)
        {
            float symmetric_value = get_symmetric_value(input);
            ret                   = info.add_literal(migraphx::literal{
                migraphx::shape{input->get_shape().type(), {1}, {0}}, {symmetric_value}});
        }

        return ret;
    }

    static instruction_ref get_zero_point(const instruction_ref& input,
                                          int index,
                                          const bool is_quant_conv,
                                          onnx_parser::node_info& info,
                                          const std::vector<instruction_ref>& args)
    {
        instruction_ref ret = input;
        if(args.size() > index)
        {
            // Check for type mismatch on parse
            if(input->get_shape().type() != args[index]->get_shape().type())
                MIGRAPHX_THROW("PARSE:Conv Data and Data Zero Point must have same type");

            ret = args[index];
            if(is_symmetric_zero_point(ret))
            {
                ret = gen_symmetric_literal(ret, is_quant_conv, info);
            }
        }
        else
        {
            ret = gen_symmetric_literal(ret, is_quant_conv, info);
        }

        return ret;
    }

    static bool is_symmetric_zero_point(instruction_ref zp)
    {
        if(not zp->can_eval())
            return false;

        float symmetric_value = get_symmetric_value(zp);

        bool all_zeros = false;
        zp->eval().visit([&](auto z) {
            all_zeros = std::all_of(
                z.begin(), z.end(), [&](auto val) { return float_equal(val, symmetric_value); });
        });
        return all_zeros;
    }

    static migraphx::operation
    qparam_broadcast_op(instruction_ref qparam, std::vector<std::size_t> lens, std::size_t axis)
    {
        if(qparam->get_shape().elements() == 1)
        {
            return migraphx::make_op("multibroadcast", {{"out_lens", lens}});
        }
        return migraphx::make_op("broadcast", {{"out_lens", lens}, {"axis", axis}});
    }

    static instruction_ref handle_quant_bias(const operation& op,
                                             const instruction_ref& input,
                                             const instruction_ref& x,
                                             const instruction_ref& weights,
                                             const instruction_ref& x_zp,
                                             const instruction_ref& w_zp,
                                             onnx_parser::node_info& info)
    {
        // to handle the bias, apply the following transformation:
        // conv(x-x_zp,w-w_zp) = conv(x,w) - conv(x_zp,w) - conv(x,w_zp) + conv(x_zp,w_zp)
        instruction_ref ret = input;

        // multibroadcast (or broadcast) zero points according to spec
        // x_zp should be a scalar or literal with one element
        // w_zp can be either a single element or a 1d tensor with size out_channels
        migraphx::operation x_zp_bc =
            migraphx::make_op("multibroadcast", {{"out_lens", x->get_shape().lens()}});
        migraphx::operation w_zp_bc = qparam_broadcast_op(w_zp, weights->get_shape().lens(), 0);

        if(not is_symmetric_zero_point(x_zp))
        {
            auto x_zp_mb  = info.add_instruction(x_zp_bc, x_zp);
            auto out_zp_1 = info.add_instruction(op, x_zp_mb, weights);
            ret           = info.add_common_op("sub", ret, out_zp_1);
        }

        if(not is_symmetric_zero_point(w_zp))
        {
            auto w_zp_mb  = info.add_instruction(w_zp_bc, w_zp);
            auto out_zp_2 = info.add_instruction(op, x, w_zp_mb);
            ret           = info.add_common_op("sub", ret, out_zp_2);
        }

        if(not(is_symmetric_zero_point(x_zp)) and not(is_symmetric_zero_point(w_zp)))
        {
            auto x_zp_mb = info.add_instruction(x_zp_bc, x_zp);
            auto w_zp_mb = info.add_instruction(w_zp_bc, w_zp);

            auto out_zp_3 = info.add_instruction(op, x_zp_mb, w_zp_mb);

            ret = info.add_common_op("add", ret, out_zp_3);
        }
        return ret;
    }

    static void handle_quant_inputs(const bool is_quant_conv,
                                    instruction_ref& input,
                                    instruction_ref& weights,
                                    instruction_ref& input_zp,
                                    instruction_ref& weight_zp,
                                    onnx_parser::node_info& info)
    {
        if(not is_quant_conv)
            return;

        auto input_type  = input->get_shape().type();
        auto weight_type = weights->get_shape().type();

        // Handle uint8 bias and input shifts
        instruction_ref offset_op;
        if(((input_type == migraphx::shape::uint8_type) or
            (weight_type == migraphx::shape::uint8_type)))
        {
            offset_op = info.add_literal(
                migraphx::literal{migraphx::shape{migraphx::shape::half_type}, {-128}});
        }

        if(input_type == migraphx::shape::uint8_type)
        {
            shift_input_and_bias(
                info, offset_op, (not is_symmetric_zero_point(input_zp)), input, input_zp);
        }

        if(weight_type == migraphx::shape::uint8_type)
        {
            shift_input_and_bias(
                info, offset_op, (not is_symmetric_zero_point(weight_zp)), weights, weight_zp);
        }
    }

    bool is_dynamic(const shape& s) const
    {
        return s.dynamic() and
               std::any_of(s.dyn_dims().begin() + 2, s.dyn_dims().end(), [](const auto& dyn_dim) {
                   return not dyn_dim.is_fixed();
               });
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto x       = args[0];
        auto weights = args[1];

        if(opd.onnx_name == "NhwcConv")
        {
            x       = from_nhwc(info, x);
            weights = from_nhwc(info, weights);
        }

        auto x_shape = x->get_shape();
        auto w_shape = weights->get_shape();
        auto in_lens = x_shape.max_lens();
        assert(in_lens.size() > 2);
        auto kdims = in_lens.size() - 2;

        // Goes into parser START 
        // ensure pads available only when auto_pad is "NOT_SET"
        check_padding_mode(info, opd.onnx_name);

        std::vector<std::size_t> strides(kdims, 1);
        if(contains(info.attributes, "strides"))
        {
            auto&& attr = info.attributes["strides"].ints();
            strides.assign(attr.begin(), attr.end());
        }

        std::vector<std::size_t> dilations(kdims, 1);
        if(contains(info.attributes, "dilations"))
        {
            auto&& attr = info.attributes["dilations"].ints();
            dilations.assign(attr.begin(), attr.end());
        }

        std::vector<int64_t> paddings;
        if(contains(info.attributes, "pads"))
        {
            auto&& attr = info.attributes["pads"].ints();
            paddings.assign(attr.begin(), attr.end());
        }

        int group = 1;
        if(contains(info.attributes, "group"))
        {
            group = parser.parse_value(info.attributes.at("group")).at<int>();
        }

        std::string auto_pad = "NOTSET";
        if(contains(info.attributes, "auto_pad"))
        {
            auto_pad = to_upper(info.attributes["auto_pad"].s());
        }
        // Goes into parser END 

        // Goes into builder
        if(strides.empty())
        {
            strides.resize(kdims);
            std::fill_n(strides.begin(), kdims, 1);
        }
        else
        {
            check_attr_sizes(kdims, strides.size(), "PARSE_CONV: inconsistent strides");
        }

        if(dilations.empty())
        {
            dilations.resize(kdims);
            std::fill_n(dilations.begin(), kdims, 1);
        }
        else
        {
            check_attr_sizes(kdims, dilations.size(), "PARSE_CONV: inconsistent dilations");
        }

        if(paddings.empty())
        {
            paddings.resize(kdims);
            std::fill_n(paddings.begin(), kdims, 0);
        }
        else if(paddings.size() != kdims and paddings.size() != 2 * kdims)
        {
            MIGRAPHX_THROW("PARSE_CONV: inconsistent paddings k-dims: " + std::to_string(kdims) +
                           " attribute size: " + std::to_string(paddings.size()));
        }

        op::padding_mode_t padding_mode = op::padding_mode_t::default_;
        if(contains(auto_pad, "SAME"))
        {
            if(is_dynamic(x_shape) or is_dynamic(w_shape))
            {
                // must calculate "same" padding with input shape data
                padding_mode = contains(auto_pad, "SAME_UPPER") ? op::padding_mode_t::same_upper
                                                                : op::padding_mode_t::same_lower;
            }
            else
            {
                // kernel shape will be fixed, so max_lens() == min_len() for kernel lengths
                auto weight_lens = weights->get_shape().max_lens();
                std::vector<std::size_t> k_lens(weight_lens.begin() + 2, weight_lens.end());
                cal_auto_padding_size(auto_pad, strides, k_lens, dilations, in_lens, paddings);
            }
        }

        auto op                = make_op(opd.op_name);
        auto values            = op.to_value();
        values["stride"]       = strides;
        values["dilation"]     = dilations;
        values["padding"]      = paddings;
        values["group"]        = group;
        values["padding_mode"] = padding_mode;
        op.from_value(values);

        // Handle quant_conv residuals between input/weights to avoid overflow
        if(opd.op_name == "quant_convolution")
        {
            auto x_zp = get_zero_point(x, 2, true, info, args);
            auto w_zp = get_zero_point(weights, 3, true, info, args);
            handle_quant_inputs(true, x, weights, x_zp, w_zp, info);
            auto conv = info.add_instruction(op, x, weights);
            return handle_quant_bias(op, conv, x, weights, x_zp, w_zp, info);
        }
        else
        {
            // Handle Convolution case with bias to output
            auto conv = info.add_instruction(op, x, weights);
            return info.add_bias(args, conv, 1);
        }

        return ret;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
