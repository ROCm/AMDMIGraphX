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

    /**
    * Makes a literal with the minimum or maximum of the input_shape's type.
    * `use_min=true` sets it to the minimum of the type, maximum otherwise. 
    */
    instruction_ref
    make_type_limit(const shape& input_shape, onnx_parser::node_info& info, bool use_min) const
    {
        instruction_ref result;
        input_shape.visit_type(
            [&](auto as) { result = info.add_literal(literal{use_min ? as.min() : as.max()}); });
        return result;
    }

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
            if(contains(info.attributes, "min"))
            {
                float min_val = parser.parse_value(info.attributes.at("min")).at<float>();
                min_arg       = info.add_literal(
                    migraphx::literal{migraphx::shape{input_shape.type()}, {min_val}});
            }
            else
            {
                min_arg = make_type_limit(input_shape, info, true);
            }

            if(contains(info.attributes, "max"))
            {
                float max_val = parser.parse_value(info.attributes.at("max")).at<float>();
                max_arg       = info.add_literal(
                    migraphx::literal{migraphx::shape{input_shape.type()}, {max_val}});
            }
            else
            {
                max_arg = make_type_limit(input_shape, info, false);
            }
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
                min_arg = make_type_limit(input_shape, info, true);
            }

            if(args.size() > 2 and not args[2]->get_shape().lens().empty())
            {
                max_arg = args[2];
            }
            else
            {
                max_arg = make_type_limit(input_shape, info, false);
            }
        }

        return op::builder::add("clip", *info.mod, {input, min_arg, max_arg}).at(0);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
