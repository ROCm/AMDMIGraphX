/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/op/builder/insert.hpp>
#include <migraphx/onnx/op_parser.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_gelu : op_parser<parse_gelu>
{
    std::vector<op_desc> operators() const { return {{"BiasGelu"}, {"FastGelu"}, {"Gelu"}}; }
    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        std::string approximate = "none";
        auto x                  = args[0];
        auto x_type             = x->get_shape().type();
        auto fast               = false;
        if(not is_type_float(x_type))
        {
            MIGRAPHX_THROW("PARSE_GELU: input tensor is not a floating type");
        }

        if(contains(info.attributes, "approximate"))
        {
            approximate = info.attributes.at("approximate").s();
        }

        if(opd.onnx_name == "FastGelu")
        {
            if(x_type == migraphx::shape::double_type)
            {
                MIGRAPHX_THROW("PARSE_GELU: FastGelu can't accept input with double precision");
            }

            // FastGelu uses tanh approximation
            approximate = "tanh";
            fast        = true;
        }

        if(args.size() > 1 and args.at(1)->name() != "undefined")
        {
            auto y      = args[1];
            auto y_type = y->get_shape().type();
            if(y_type != x_type)
            {
                MIGRAPHX_THROW("PARSE_GELU: mismatching input tensor types");
            }
            x = info.add_common_op("add", x, y);
        }

        if(approximate == "tanh")
        {
            return op::builder::add("gelu_tanh", *info.mod, {x}, {{"fast", fast}}).at(0);
        }
        else
        {
            return op::builder::add("gelu_erf", *info.mod, {x}, {}).at(0);
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
