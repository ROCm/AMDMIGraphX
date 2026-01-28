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
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_concat_from_sequence : op_parser<parse_concat_from_sequence>
{
    std::vector<op_desc> operators() const
    {
        return {{"ConcatFromSequence"}};
    }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        std::vector<instruction_ref> inputs = std::move(args);

        int64_t axis = 0;
        if(contains(info.attributes, "axis"))
        {
            axis = info.attributes.at("axis").i();
        }

        int64_t new_axis = 0;
        if(contains(info.attributes, "new_axis"))
        {
            new_axis = info.attributes.at("new_axis").i();
        }

        if(new_axis == 1)
        {
            std::vector<instruction_ref> unsqueezed_inputs;
            unsqueezed_inputs.reserve(inputs.size());
            for(auto& input : inputs)
            {
                unsqueezed_inputs.push_back(
                    info.add_instruction(make_op("unsqueeze", {{"axes", {axis}}}), input)
                );
            }
            inputs = unsqueezed_inputs;
        }

        return info.add_instruction(make_op("concat", {{"axis", axis}}), inputs);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
