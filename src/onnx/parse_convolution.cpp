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

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto op       = make_op(opd.op_name);
        auto values   = op.to_value();
        auto l0       = args[0];
        auto weights  = args[1];
        auto l0_shape = l0->get_shape();
        auto w_shape  = weights->get_shape();
        auto in_lens  = l0_shape.max_lens();
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
            if(l0_shape.dynamic())
            {
                auto dyn_dims = l0_shape.dyn_dims();
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
        auto l0_zp = l0;
        auto w_zp  = weights;

        op.from_value(values);
        if(op.name() == "quant_convolution")
        {
            if(args.size() > 2)
            {
                l0_zp = args[2];
                if(l0_zp->get_shape().type() != l0_shape.type())
                {
                    MIGRAPHX_THROW(
                        "PARSE: ConvInteger Data and Data Zero Point must have same type");
                }

                l0_zp = info.add_common_op("sub", l0, l0_zp);

                if(args.size() > 3)
                {
                    w_zp = args[3];
                    if(w_zp->get_shape().type() != w_shape.type())
                    {
                        MIGRAPHX_THROW(
                            "PARSE: ConvInteger Weight and Weight Zero Point must have same type");
                    }

                    w_zp = info.add_common_op("sub", weights, w_zp);
                }

                ret = info.add_instruction(op, l0_zp, w_zp);
            }
        }
        else
        {
            auto l1 = info.add_instruction(op, l0, args[1]);
            ret     = info.add_bias(args, l1, 1);
        }
        return ret;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
