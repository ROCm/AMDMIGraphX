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
#include <migraphx/onnx/op_parser.hpp>
#include <migraphx/onnx/checks.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/sym.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_topk : op_parser<parse_topk>
{
    std::vector<op_desc> operators() const { return {{"TopK"}}; }

    std::vector<instruction_ref> parse(const op_desc& /*opd*/,
                                       const onnx_parser& parser,
                                       onnx_parser::node_info info,
                                       std::vector<instruction_ref> args) const
    {
        bool largest = true;
        if(contains(info.attributes, "largest"))
        {
            largest = static_cast<bool>(info.attributes.at("largest").i());
        }

        int64_t axis = -1;
        if(contains(info.attributes, "axis"))
        {
            axis = parser.parse_value(info.attributes.at("axis")).at<int>();
        }

        // opset-1 form: `k` is an attribute. Synthesize a constant `k` input so the topk
        // operator always has (x, k) inputs.
        if(args.size() == 1)
        {
            int64_t k = 0;
            if(contains(info.attributes, "k"))
            {
                k = info.attributes.at("k").i();
            }
            const shape k_shape{shape::int64_type, {1}};
            auto k_lit    = info.add_literal(literal{k_shape, {k}});
            auto topk_ret = info.add_instruction(
                make_op("topk", {{"k", k}, {"axis", axis}, {"largest", largest}}), args.at(0), k_lit);

            auto ret_val = info.add_instruction(make_op("get_tuple_elem", {{"index", 0}}), topk_ret);
            auto ret_ind = info.add_instruction(make_op("get_tuple_elem", {{"index", 1}}), topk_ret);
            return {ret_val, ret_ind};
        }

        // opset-10+ form: `k` is a runtime input.
        auto arg_k = args.at(1)->eval();
        if(not arg_k.empty())
        {
            // Constant `k`: use its value for the attribute; topk output is already the exact size.
            int64_t k     = arg_k.at<int>();
            auto topk_ret = info.add_instruction(
                make_op("topk", {{"k", k}, {"axis", axis}, {"largest", largest}}), args.at(0), args.at(1));

            auto ret_val = info.add_instruction(make_op("get_tuple_elem", {{"index", 0}}), topk_ret);
            auto ret_ind = info.add_instruction(make_op("get_tuple_elem", {{"index", 1}}), topk_ret);
            return {ret_val, ret_ind};
        }

        // Variable (data-dependent) `k`: run topk over the whole axis dimension, then slice the
        // outputs down to the runtime `k` using a symbolic dimension.
        auto input_shape = args.at(0)->get_shape();
        auto norm_axis   = axis < 0 ? axis + input_shape.ndim() : axis;
        int64_t k        = input_shape.max_lens().at(norm_axis);

        auto topk_ret = info.add_instruction(
            make_op("topk", {{"k", k}, {"axis", axis}, {"largest", largest}}), args.at(0), args.at(1));

        auto ret_val = info.add_instruction(make_op("get_tuple_elem", {{"index", 0}}), topk_ret);
        auto ret_ind = info.add_instruction(make_op("get_tuple_elem", {{"index", 1}}), topk_ret);

        auto k_var = shape::dynamic_dimension{sym::var(info.name)};
        ret_val    = info.add_instruction(make_op("slice",
                                               {{"axes", {axis}},
                                                {"starts", {0}},
                                                {"ends", value::array{to_value(k_var)}},
                                                {"mode", "ends_input"}}),
                                       ret_val,
                                       args.at(1));
        ret_ind    = info.add_instruction(make_op("slice",
                                               {{"axes", {axis}},
                                                {"starts", {0}},
                                                {"ends", value::array{to_value(k_var)}},
                                                {"mode", "ends_input"}}),
                                       ret_ind,
                                       args.at(1));

        return {ret_val, ret_ind};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
