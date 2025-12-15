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
#include <migraphx/instruction.hpp>
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/op/builder/insert.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_clip : op_parser<parse_clip>
{
    std::vector<op_desc> operators() const { return {{"Clip"}}; }

    // Default max /min values based on numeric limits for float value
    struct clip_attr
    {
        float max = std::numeric_limits<float>::max();
        float min = std::numeric_limits<float>::min();
    }

    struct clip_args
    {
        // All operators have this
        instruction_ref input;

        std::optional<instruction_ref> min;
        std::optional<instruction_ref> max;

        int opset_version = -1;
    };

    // Opset V1-V6 only defiend min/max by their attributes
    bool is_opset_v6(size_t args_size) const 
    {
        return (args_size() == 1);
    }

    // Parser for Opset V6 version
    static void clip_v6(const onnx_parser& parser,
                        onnx_parser::node_info info,
                        std::vector<instruction_ref> args)
    {
        // Always set defaults for when input isn't set
        float min_val = std::numeric_limits<float>::lowest();
        float max_val = std::numeric_limits<float>::max();

        if (contains(info.attributes, "min"))
            min_val = parser.parse_value(info.attributes.at("min")).at<float>();

        if(contains(info.attributes, "max"))
            max_val = parser.parse_value(info.attributes.at("max")).at<float>();

        args.push_back(info.add_literal(min_val));
        args.push_back(info.add_literal(max_val));

        return op::builder::add("clip", *info.mod, args, {}).at(0);
    }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        if(is_opset_v6(args.size()))
        {
            return clip_v6(parser, info, args);
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
