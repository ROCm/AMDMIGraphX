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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

instruction_ref parse_gelu_no_approx(onnx_parser::node_info info, instruction_ref x)
{
    auto x_shape  = x->get_shape();
    auto half     = info.add_literal(migraphx::literal{x_shape, {0.5f}});
    auto one      = info.add_literal(migraphx::literal{x_shape, {1.0f}});
    auto sqrt2    = info.add_literal(migraphx::literal{x_shape, {static_cast<float>(M_SQRT2)}});
    auto mul_half = info.add_instruction(migraphx::make_op("mul"), x, half);
    auto div      = info.add_instruction(migraphx::make_op("div"), x, sqrt2);
    auto erf      = info.add_instruction(migraphx::make_op("erf"), div);
    auto add_one  = info.add_instruction(migraphx::make_op("add"), erf, one);
    return info.add_instruction(migraphx::make_op("mul"), mul_half, add_one);
}

instruction_ref parse_gelu_tanh_approx(onnx_parser::node_info info, instruction_ref x)
{
    auto x_shape    = x->get_shape();
    auto fit_const  = info.add_literal(migraphx::literal{x_shape, {0.044708251953125}});
    auto sqrt_2_rpi = info.add_literal(migraphx::literal{x_shape, {0.7978515625}});
    auto one        = info.add_literal(migraphx::literal{x_shape, {1.0f}});
    auto one_half   = info.add_literal(migraphx::literal{x_shape, {0.5f}});
    auto three      = info.add_literal(migraphx::literal{x_shape, {3.0f}});
    auto pow0       = info.add_instruction(migraphx::make_op("pow"), {x, three});
    auto mul0       = info.add_instruction(migraphx::make_op("mul"), {pow0, fit_const});
    auto add0       = info.add_instruction(migraphx::make_op("add"), {mul0, x});
    auto mul1       = info.add_instruction(migraphx::make_op("mul"), {add0, sqrt_2_rpi});
    auto tanh0      = info.add_instruction(migraphx::make_op("tanh"), mul1);
    auto add1       = info.add_instruction(migraphx::make_op("add"), {tanh0, one});
    auto mul2       = info.add_instruction(migraphx::make_op("mul"), {x, one_half});
    return info.add_instruction(migraphx::make_op("mul"), {add1, mul2});
}

struct parse_gelu : op_parser<parse_gelu>
{
    std::vector<op_desc> operators() const { return {{"Gelu"}}; }
    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        std::string approximate = "none";
        auto x                  = args[0];
        if(not is_type_float(x->get_shape().type()))
        {
            MIGRAPHX_THROW("PAERSE_GELU: input tensor is not a floating type");
        }

        if(contains(info.attributes, "approximate"))
        {
            approximate = info.attributes["approximate"].s();
        }
        if (approximate == "tanh")
        {
            return parse_gelu_tanh_approx(info, x);
        }
        else
        {
            return parse_gelu_no_approx(info, x);
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
