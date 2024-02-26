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
#include <migraphx/onnx/gelu.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

bool is_fast_gelu_input_type(shape::type_t input_type)
{
    const std::vector<migraphx::shape::type_t> fast_gelu_type{shape::float_type, shape::half_type};
    return std::any_of(fast_gelu_type.begin(), fast_gelu_type.end(), [&](auto type) {
        return type == input_type;
    });
}

struct parse_fastgelu : op_parser<parse_fastgelu>
{
    std::vector<op_desc> operators() const { return {{"FastGelu"}}; }
    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        const std::vector<migraphx::shape::type_t> fast_gelu_type{shape::float_type,
                                                                  shape::half_type};
        auto x      = args[0];
        auto x_type = x->get_shape().type();
        if(not is_fast_gelu_input_type(x_type))
        {
            MIGRAPHX_THROW("PARSE_FASTGELU: input tensor `x` is not a float or half type");
        }
        if(args.size() > 1 and args.at(1)->name() != "undefined")
        {
            auto y      = args[1];
            auto y_type = y->get_shape().type();
            if(not is_fast_gelu_input_type(y_type))
            {
                MIGRAPHX_THROW("PARSE_FASTGELU: input tensor `bias` is not a float or half type");
            }
            x = info.add_common_op("add", x, y);
        }

        auto x_lens = x->get_shape().lens();
        // FastGelu equation from
        // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftfastgelu
        // Y=0.5X(1+tanh(0.797885X+0.035677XXX))
        auto const1 = info.add_literal(migraphx::literal{migraphx::shape{x_type}, {0.797885}});
        auto const2 = info.add_literal(migraphx::literal{migraphx::shape{x_type}, {0.035677}});
        auto one    = info.add_literal(migraphx::literal{migraphx::shape{x_type}, {1.0f}});
        auto half   = info.add_literal(migraphx::literal{migraphx::shape{x_type}, {0.5f}});
        auto three  = info.add_literal(migraphx::literal{migraphx::shape{x_type}, {3.0f}});
        // 0.035677XXX
        auto three_mbcast = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x_lens}}), three);
        auto pow0          = info.add_instruction(migraphx::make_op("pow"), {x, three_mbcast});
        auto const2_mbcast = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x_lens}}), const2);
        auto mul0 = info.add_instruction(migraphx::make_op("mul"), {pow0, const2_mbcast});

        // 0.797885X+0.035677XXX
        auto const1_mbcast = info.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", x_lens}}), const1);
        auto mul1 = info.add_instruction(migraphx::make_op("mul"), {const1_mbcast, x});
        auto add1 = info.add_instruction(migraphx::make_op("add"), {mul0, mul1});

        // 1+tanh(0.797885X+0.035677XXX)
        auto tanh0 = info.add_instruction(migraphx::make_op("tanh"), add1);
        auto one_mbcast =
            info.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", x_lens}}), one);
        auto add2 = info.add_instruction(migraphx::make_op("add"), {tanh0, one_mbcast});

        // 0.5X(1+tanh(0.797885X+0.035677XXX))
        auto half_mbcast =
            info.add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", x_lens}}), half);
        auto mul2 = info.add_instruction(migraphx::make_op("mul"), {x, half_mbcast});
        return info.add_instruction(migraphx::make_op("mul"), {add2, mul2});
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
