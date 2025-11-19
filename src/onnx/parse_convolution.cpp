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
#include <migraphx/op/builder/insert.hpp>

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

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        if(opd.onnx_name == "NhwcConv")
        {
            auto x       = args[0];
            auto weights = args[1];

            args[0] = from_nhwc(info, x);
            args[1] = from_nhwc(info, weights);
        }

        // ensure pads available only when auto_pad is "NOT_SET"
        check_padding_mode(info, opd.onnx_name);

        value options = {};
        if(contains(info.attributes, "strides"))
        {
            const auto& attr = info.attributes["strides"].ints();
            std::vector<std::size_t> strides{attr.begin(), attr.end()};
            options.insert({"strides", strides});
        }

        if(contains(info.attributes, "dilations"))
        {
            const auto& attr = info.attributes["dilations"].ints();
            std::vector<std::size_t> dilations{attr.begin(), attr.end()};
            options.insert({"dilations", dilations});
        }

        if(contains(info.attributes, "pads"))
        {
            const auto& attr = info.attributes["pads"].ints();
            std::vector<int64_t> paddings{attr.begin(), attr.end()};
            options.insert({"paddings", paddings});
        }

        if(contains(info.attributes, "group"))
        {
            const auto group = parser.parse_value(info.attributes.at("group")).at<int>();
            options.insert({"group", group});
        }

        if(contains(info.attributes, "auto_pad"))
        {
            const auto auto_pad = to_upper(info.attributes["auto_pad"].s());
            options.insert({"auto_pad", auto_pad});
        }

        auto ret = op::builder::add(opd.op_name, *info.mod, args, options).at(0);

        if(opd.onnx_name == "NhwcConv")
        {
            ret = to_nhwc(info, ret);
        }

        return ret;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
