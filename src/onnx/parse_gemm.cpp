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

#include <migraphx/ranges.hpp>
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/onnx/op_parser.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_gemm : op_parser<parse_gemm>
{
    std::vector<op_desc> operators() const { return {{"Gemm"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          const std::vector<instruction_ref>& args) const
    {
        value options = {};

        if(contains(info.attributes, "alpha"))
        {
            const float alpha = parser.parse_value(info.attributes.at("alpha")).at<float>();
            options.insert({"alpha", alpha});
        }
        if(contains(info.attributes, "beta"))
        {
            const float beta = parser.parse_value(info.attributes.at("beta")).at<float>();
            options.insert({"beta", beta});
        }
        if(contains(info.attributes, "transA"))
        {
            const bool trans_a = parser.parse_value(info.attributes.at("transA")).at<bool>();
            options.insert({"transA", trans_a});
        }
        if(contains(info.attributes, "transB"))
        {
            const bool trans_b = parser.parse_value(info.attributes.at("transB")).at<bool>();
            options.insert({"transB", trans_b});
        }

        return op::builder::add("gemm", *info.mod, args, options).at(0);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
