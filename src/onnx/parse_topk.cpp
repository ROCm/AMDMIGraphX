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
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/tune_axis.hpp>
#include <migraphx/value.hpp>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_topk : op_parser<parse_topk>
{
    std::vector<op_desc> operators() const { return {{"TopK"}}; }

    static std::vector<instruction_ref> add_topk_and_gets(const onnx_parser::node_info& info,
                                                          const std::vector<instruction_ref>& args,
                                                          int64_t k,
                                                          int64_t axis,
                                                          bool largest)
    {
        auto topk_ret = info.add_instruction(
            make_op("topk", {{"k", k}, {"axis", axis}, {"largest", largest}}), args.at(0));
        auto ret_val = info.add_instruction(make_op("get_tuple_elem", {{"index", 0}}), topk_ret);
        auto ret_ind = info.add_instruction(make_op("get_tuple_elem", {{"index", 1}}), topk_ret);
        return {ret_val, ret_ind};
    }

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

        // opset-1 form: `k` is an ONNX attribute. Go directly into MIGX's topk operator.
        if(args.size() == 1)
        {
            int64_t k = 0;
            if(contains(info.attributes, "k"))
            {
                k = info.attributes.at("k").i();
            }
            return add_topk_and_gets(info, args, k, axis, largest);
        }

        // opset-10+ form: `k` is a runtime input.
        auto k_ins = args.at(1);
        auto arg_k = k_ins->eval();
        if(not arg_k.empty())
        {
            // constant `k` value
            int64_t k = arg_k.at<int>();
            return add_topk_and_gets(info, args, k, axis, largest);
        }
        // Variable `k` is supported only for static or symbolic data. Run TopK at the axis
        // maximum, then slice both outputs to the runtime `k`.
        const auto& input_shape = args.at(0)->get_shape();
        if(input_shape.dynamic() and not input_shape.symbolic())
            MIGRAPHX_THROW("PARSE_TOPK: a runtime `k` needs a static or symbolic data shape; "
                           "parse with symbolic shapes enabled");

        auto norm_axis = tune_axis(input_shape.ndim(), axis, "TopK");
        int64_t max_k  = input_shape.max_lens().at(norm_axis);
        auto outputs   = add_topk_and_gets(info, args, max_k, norm_axis, largest);
        auto k_end     = sym::var(info.name, {0, max_k});
        auto starts    = info.add_literal(literal{{shape::int64_type, {1}}, {0}});
        auto slice     = make_op(
            "dyn_slice",
            {{"axes", {norm_axis}}, {"starts", {0}}, {"ends", value::array{to_value(k_end)}}});
        std::transform(outputs.begin(), outputs.end(), outputs.begin(), [&](auto output) {
            return info.add_instruction(slice, output, starts, k_ins);
        });
        return outputs;
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
