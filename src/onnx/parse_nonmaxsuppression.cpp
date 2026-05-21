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

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_USE_DYNAMIC_NMS);

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_nonmaxsuppression : op_parser<parse_nonmaxsuppression>
{
    std::vector<op_desc> operators() const { return {{"NonMaxSuppression", "nonmaxsuppression"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        auto op = parser.load(opd.op_name, info);
        auto nms_ins = info.add_instruction(op, args);
        // variable ends input slice to handle dynamic shape output
        auto indices = info.add_instruction(make_op("get_tuple_elem", {{"index", 0}}), nms_ins);
        if(enabled(MIGRAPHX_USE_DYNAMIC_NMS{}))
        {
            return indices;
        }
        else
        {
            auto num_selected = info.add_instruction(make_op("get_tuple_elem", {{"index", 1}}), nms_ins);
            auto slice_ins = info.add_instruction(make_op("slice", {{"axes", {0}}, {"starts", {0}}}), indices, num_selected);
            return slice_ins;
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
