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
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/tune_axis.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_dequantizelinear : op_parser<parse_dequantizelinear>
{
    std::vector<op_desc> operators() const { return {{"DequantizeLinear"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& /*parser*/,
                          const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args) const
    {
        int axis = 1;
        if(contains(info.attributes, "axis"))
            axis = info.attributes.at("axis").i();

        auto input_lens = args[0]->get_shape().lens();
        auto n_dim      = input_lens.size();

        instruction_ref x_scale;
        if(args[1]->get_shape().elements() != 1)
        {
            auto tuned_axis = tune_axis(n_dim, axis, opd.op_name);
            x_scale         = info.add_instruction(
                make_op("broadcast", {{"axis", tuned_axis}, {"out_lens", input_lens}}), args[1]);
        }
        else
        {
            x_scale = info.add_instruction(make_op("multibroadcast", {{"out_lens", input_lens}}),
                                           args[1]);
        }

        if(args.size() == 3)
        {
            auto x_zero_point = args[2];
            if(x_zero_point->get_shape().elements() != 1)
            {
                auto tuned_axis = tune_axis(n_dim, axis, opd.op_name);
                x_zero_point    = info.add_instruction(
                    make_op("broadcast", {{"axis", tuned_axis}, {"out_lens", input_lens}}),
                    x_zero_point);
            }
            else
            {
                x_zero_point = info.add_instruction(
                    make_op("multibroadcast", {{"out_lens", input_lens}}), x_zero_point);
            }

            return info.add_instruction(
                make_op("dequantizelinear"), args[0], x_scale, x_zero_point);
        }

        return info.add_instruction(make_op("dequantizelinear"), args[0], x_scale);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
