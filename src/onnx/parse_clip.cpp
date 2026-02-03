/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2026 Advanced Micro Devices, Inc. All rights reserved.
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

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto input       = args.at(0);
        auto input_shape = input->get_shape();

        instruction_ref min_arg;
        instruction_ref max_arg;

        // Check if older opset (attributes for min/max)
        if(contains(info.attributes, "min") or contains(info.attributes, "max"))
        {
            // Opset 1-6: min/max are attributes
            float min_val = std::numeric_limits<float>::lowest();
            float max_val = std::numeric_limits<float>::max();

            if(contains(info.attributes, "min"))
                min_val = parser.parse_value(info.attributes.at("min")).at<float>();

            if(contains(info.attributes, "max"))
                max_val = parser.parse_value(info.attributes.at("max")).at<float>();

            // Per ONNX spec: if min > max, set min = max
            if(min_val > max_val)
                min_val = max_val;

            min_arg = info.add_literal(migraphx::literal{migraphx::shape{input_shape.type()}, {min_val}});
            max_arg = info.add_literal(migraphx::literal{migraphx::shape{input_shape.type()}, {max_val}});
        }
        else
        {
            // Opset 11+: min/max are optional inputs
            // args[0] = input
            // args[1] = min (optional, may be empty)
            // args[2] = max (optional, may be empty)

            if(args.size() > 1 and not args[1]->get_shape().lens().empty())
            {
                min_arg = args[1];
            }
            else
            {
                input_shape.visit_type([&](auto as) {
                    min_arg = info.add_literal(literal{as.min()});
                });
            }

            if(args.size() > 2 and not args[2]->get_shape().lens().empty())
            {
                max_arg = args[2];
            }
            else
            {
                input_shape.visit_type([&](auto as) {
                    max_arg = info.add_literal(literal{as.max()});
                });
            }
        }

        return op::builder::add("clip", *info.mod, {input, min_arg, max_arg}, {}).at(0);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
