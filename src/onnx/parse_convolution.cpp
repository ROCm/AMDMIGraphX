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
        auto x       = args[0];
        auto in_lens = x->get_shape().max_lens();
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

        auto op_name = opd.op_name == "quant_convolution" ? "convolution_integer" : "convolution";
        return op::builder::add(op_name,
                                *info.mod,
                                args,
                                {{"strides", strides},
                                 {"auto_pad", auto_pad},
                                 {"dilations", dilations},
                                 {"paddings", paddings},
                                 {"group", group}})
            .at(0);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
