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

    struct clip_args
    {
        // All operators have this
        instruction_ref input;

        std::optional<instruction_ref> min;
        std::optional<instruction_ref> max;
    };

    std::vector<instruction_ref> get_args(clip_args& arg)
    {
        return {arg.input, arg.min.value(), arg.max.value()};
    }

    static std::optional<instruction_ref>
    check_type_and_shape(size_t index, shape::type_t type,
                         const std::vector<instruction_ref>& args)
    {

        if (args.size() > index)
        {
            std::optional<instruction_ref> ref = args.at(index);
            auto shape_scalar = ref.value().get_shape().scalar();
            auto ref_type = ref.value().get_shape().type();

            if(not shape_scaler)
               MIGRAPHX_THROW("Invalid input for CLIP must be scalar");

            if (ref_type != type)
               MIGRAPHX_THROW("Invalid input type for clip min/max must match input type");

            return ref;
        }
        return {};
    }

    static void handle_limits(onnx_parser::node_info info,
                              clip_args& clip_parser)
    {
        // Set default if types/inputs aren't set
        // Try to see if we can fold limit value during parse

        // min
        if(not clip_parser.min.has_value())
        {  
           clip_parser.min = info.add_literal(std::numeric_limits<float>::lowest());
        }
        else
        {
           if(clip_parser.min->value()->front()->can_eval())
           {
              clip_parser.min = info.add_literal(clip_parser.min.value()->front()->eval());
           }
        }

        // max
        if(not clip_parser.max.has_value())
        {
           clip_parser.max = info.add_literal(std::numeric_limits<float>::max());
        }
        else
        {
           if(clip_parser.max->value()->front()->can_eval())
           {
             clip_parser.max = info.add_literal(clip_parser.max.value()->front()->eval());
           }
        }
    }

    // Parser for Opset 11, 12, 13 
    static void clip_v_11_12_13(onnx_parser::node_info info,
                                const std::vector<instruction_ref>& args)
    {   
        clip_args clip_parser;

        clip_parser.input = args.at(0);
        auto input_type = clip_parser.input.value().get_shape().type();

        clip_parser.min = check_type_and_shape(1, input_type, args);
        clip_parser.max = check_type_and_shape(2, input_type, args);

        handle_limits(clip_parser);

        return op::builder::add("clip", *info.mod, get_args(clip_parser), {}).at(0);
    }

    // Parser for Opset V6 version
    static void clip_v6(const onnx_parser& parser,
                        onnx_parser::node_info info,
                        std::vector<instruction_ref>& args)
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
        if(parser.get_opset_version() < 11)
        {
            return clip_v6(parser, info, args);
        }
        else
        {
            return clip_v_11_12_13(info, args);
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
