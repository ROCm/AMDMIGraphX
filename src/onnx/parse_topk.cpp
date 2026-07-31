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
            auto topk_ret = info.add_instruction(
                make_op("topk", {{"k", k}, {"axis", axis}, {"largest", largest}}), args.at(0));
            auto ret_val =
                info.add_instruction(make_op("get_tuple_elem", {{"index", 0}}), topk_ret);
            auto ret_ind =
                info.add_instruction(make_op("get_tuple_elem", {{"index", 1}}), topk_ret);
            return {ret_val, ret_ind};
        }

        // opset-10+ form: `k` is a runtime input. A constant `k` gives an exactly sized output,
        // so no slicing is needed.
        auto arg_k = args.at(1)->eval();
        if(not arg_k.empty())
        {
            auto topk_ret = info.add_instruction(
                make_op("topk", {{"k", arg_k.at<int64_t>()}, {"axis", axis}, {"largest", largest}}),
                args.at(0));
            auto ret_val =
                info.add_instruction(make_op("get_tuple_elem", {{"index", 0}}), topk_ret);
            auto ret_ind =
                info.add_instruction(make_op("get_tuple_elem", {{"index", 1}}), topk_ret);
            return {ret_val, ret_ind};
        }

        // Variable (data-dependent) `k`: name the runtime value with a symbol and let dyn_topk
        // describe its output as min(k, axis length). ONNX requires 1 <= k <= axis length, and
        // those bounds are what keep the resulting dimension's interval finite.
        // TODO: rewrite_topk should later turn this into topk + slice so it can run on a target.
        auto input_shape = args.at(0)->get_shape();
        auto norm_axis   = axis < 0 ? axis + input_shape.ndim() : axis;
        int64_t k_max    = input_shape.max_lens().at(norm_axis);
        auto k_var       = sym::var(info.name, {1, k_max});

        auto topk_ret = info.add_instruction(
            make_op("dyn_topk", {{"k", to_value(k_var)}, {"axis", axis}, {"largest", largest}}),
            args.at(0),
            args.at(1));
        auto ret_val = info.add_instruction(make_op("get_tuple_elem", {{"index", 0}}), topk_ret);
        auto ret_ind = info.add_instruction(make_op("get_tuple_elem", {{"index", 1}}), topk_ret);
        return {ret_val, ret_ind};
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
