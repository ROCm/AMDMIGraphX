/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/onnx/checks.hpp>
#include <migraphx/instruction.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_softmaxcrossentropyloss : op_parser<parse_softmaxcrossentropyloss>
{
    std::vector<op_desc> operators() const
    {
        return {{"SoftmaxCrossEntropyLoss", "softmaxcrossentropyloss"}};
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {

        // default in Onnx spec is mean setting
        std::string reduction = "mean";

        if(contains(info.attributes, "reduction"))
        {
            reduction = parser.parse_value(info.attributes.at("reduction")).at<std::string>();
        }

        // ignore_index is optional attribute
        if(contains(info.attributes, "ignore_index"))
        {
            int ignore_index = parser.parse_value(info.attributes.at("ignore_index")).at<int>();

            return info.add_instruction(
                make_op(opd.op_name, {{"reduction", reduction}, {"ignore_index", ignore_index}}),
                args);
        }

        return info.add_instruction(make_op(opd.op_name, {{"reduction", reduction}}), args);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
