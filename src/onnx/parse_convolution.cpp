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
#include <migraphx/onnx/checks.hpp>
#include <migraphx/onnx/conv.hpp>
#include <migraphx/onnx/padding.hpp>
#include <migraphx/op/common.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_convolution : op_parser<parse_convolution>
{
    std::vector<op_desc> operators() const
    {
        return {{"Conv", "convolution"}, {"ConvInteger", "quant_convolution"}};
    }

    static instruction_ref set_bias(const instruction_ref& input,
                                    onnx_parser::node_info info,
                                    int index,
                                    const bool is_quant_conv,
                                    const std::vector<instruction_ref>& args)
    {
        instruction_ref ret = input;
        if(args.size() > index)
        {
            // Check for type mismatch on parse
            if(input->get_shape().type() != args[index]->get_shape().type())
                MIGRAPHX_THROW("PARSE:Conv Data and Data Zero Point must have same type");

            if(is_quant_conv)
                ret = info.add_common_op("sub", input, args[index]);
        }
        return ret;
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto op      = make_op(opd.op_name);
        auto values  = op.to_value();
        auto x       = args[0];
        auto weights = args[1];
        auto x_shape = x->get_shape();
        auto w_shape = weights->get_shape();
        auto in_lens = x_shape.max_lens();
        assert(in_lens.size() > 2);
        auto kdims = in_lens.size() - 2;

        // ensure pads available only when auto_pad is "NOT_SET"
        check_padding_mode(info, "CONV");

        if(contains(info.attributes, "strides"))
        {
            values["stride"].clear();
            copy(info.attributes["strides"].ints(), std::back_inserter(values["stride"]));
            check_attr_sizes(kdims, values["stride"].size(), "PARSE_CONV: inconsistent strides");
        }
        if(contains(info.attributes, "dilations"))
        {
            values["dilation"].clear();
            copy(info.attributes["dilations"].ints(), std::back_inserter(values["dilation"]));
            check_attr_sizes(
                kdims, values["dilation"].size(), "PARSE_CONV: inconsistent dilations");
        }

        std::vector<int64_t> padding;
        if(contains(info.attributes, "pads"))
        {
            values["padding"].clear();
            copy(info.attributes["pads"].ints(), std::back_inserter(padding));
            check_attr_sizes(kdims, padding.size() / 2, "PARSE_CONV: inconsistent paddings");
        }
        if(contains(info.attributes, "auto_pad"))
        {
            bool is_same_padding = false;
            auto auto_pad        = info.attributes["auto_pad"].s();
            if(auto_pad.find("SAME") != std::string::npos)
            {
                is_same_padding = true;
            }

            // check if image shape is dynamic
            bool image_shape_dynamic = false;
            if(x_shape.dynamic())
            {
                auto dyn_dims = x_shape.dyn_dims();
                std::for_each(dyn_dims.begin() + 2, dyn_dims.end(), [&](auto dyn_dim) {
                    if(not dyn_dim.is_fixed())
                    {
                        image_shape_dynamic = true;
                    }
                });
            }

            // check if kernel shape is dynamic
            bool kernel_shape_dynamic = false;
            if(w_shape.dynamic())
            {
                auto dyn_dims = w_shape.dyn_dims();
                std::for_each(dyn_dims.begin() + 2, dyn_dims.end(), [&](auto dyn_dim) {
                    if(not dyn_dim.is_fixed())
                    {
                        kernel_shape_dynamic = true;
                    }
                });
            }

            if(is_same_padding)
            {
                if(image_shape_dynamic or kernel_shape_dynamic)
                {
                    // must calculate "same" padding with input shape data
                    bool is_same_upper     = (auto_pad.find("SAME_UPPER") != std::string::npos);
                    values["padding_mode"] = is_same_upper
                                                 ? to_value(op::padding_mode_t::same_upper)
                                                 : to_value(op::padding_mode_t::same_lower);
                }
                else
                {
                    // kernel shape will be fixed, so max_lens() == min_len() for kernel lengths
                    auto weight_lens = weights->get_shape().max_lens();
                    std::vector<std::size_t> k_lens(weight_lens.begin() + 2, weight_lens.end());
                    cal_auto_padding_size(info,
                                          values,
                                          k_lens,
                                          values["dilation"].to_vector<std::size_t>(),
                                          in_lens,
                                          padding);
                }
            }
        }
        values["padding"] = std::vector<size_t>(padding.begin(), padding.end());

        if(contains(info.attributes, "group"))
        {
            values["group"] = parser.parse_value(info.attributes.at("group")).at<int>();
        }

        recalc_conv_attributes(values, kdims);

        instruction_ref ret;
        // parse a_zero_point and b_zero_point values
        auto is_quant_conv = opd.op_name == "quant_convolution";
        auto x_zp          = set_bias(x, info, 2, is_quant_conv, args);
        auto w_zp          = set_bias(weights, info, 3, is_quant_conv, args);

        op.from_value(values);

        x_zp->debug_print();

        ret = info.add_instruction(op, x_zp, w_zp);

        // Handle Convolution case with bias to output
        if((not is_quant_conv) and (args.size() > 2))
        {
            ret = info.add_bias(args, ret, 1);
        }

        return ret;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
