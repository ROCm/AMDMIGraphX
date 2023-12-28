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
#include <migraphx/ranges.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/onnx/checks.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

instruction_ref parse_reduce_oper(const std::string& op_name,
                                  const onnx_parser& parser,
                                  onnx_parser::node_info info,
                                  std::vector<instruction_ref> args)
{
    // default to reduce over all dimensions
    std::vector<int64_t> axes;
    if(args.size() == 2)
    {
        auto arg_axes = args.at(1)->eval();
        check_arg_empty(arg_axes, "PARSE_" + op_name + ": cannot handle variable axes!");
        axes.clear();
        arg_axes.visit([&](auto s) { axes.assign(s.begin(), s.end()); });
    }
    else if(contains(info.attributes, "axes"))
    {
        axes.clear();
        auto&& attr_axes = info.attributes["axes"].ints();
        axes             = std::vector<int64_t>(attr_axes.begin(), attr_axes.end());
    }

    bool noop_with_empty_axes = false;
    if(contains(info.attributes, "noop_with_empty_axes"))
    {
        noop_with_empty_axes = static_cast<bool>(
            parser.parse_value(info.attributes.at("noop_with_empty_axes")).at<int>());
    }

    // empty axes behavior
    if(axes.empty())
    {
        if(noop_with_empty_axes)
        {
            return args.at(0);
        }
        else
        {
            axes.resize(args.front()->get_shape().ndim());
            std::iota(axes.begin(), axes.end(), 0);
        }
    }

    int keep_dims = 1;
    if(contains(info.attributes, "keepdims"))
    {
        keep_dims = parser.parse_value(info.attributes.at("keepdims")).at<int>();
    }

    if(keep_dims == 1)
    {
        return info.add_instruction(make_op(op_name, {{"axes", axes}}), args.front());
    }
    else
    {
        auto ins = info.add_instruction(make_op(op_name, {{"axes", axes}}), args.front());
        return info.add_instruction(make_op("squeeze", {{"axes", axes}}), ins);
    }
}

struct parse_reduce_op : op_parser<parse_reduce_op>
{
    std::vector<op_desc> operators() const
    {
        return {{"ReduceMax", "reduce_max"},
                {"ReduceMean", "reduce_mean"},
                {"ReduceMin", "reduce_min"},
                {"ReduceProd", "reduce_prod"},
                {"ReduceSum", "reduce_sum"}};
    }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        return parse_reduce_oper(opd.op_name, parser, std::move(info), std::move(args));
    }
};

struct parse_reduce_l1 : op_parser<parse_reduce_l1>
{
    std::vector<op_desc> operators() const { return {{"ReduceL1"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto abs_ins = info.add_instruction(make_op("abs"), args[0]);
        return parse_reduce_oper("reduce_sum", parser, std::move(info), {abs_ins});
    }
};

struct parse_reduce_l2 : op_parser<parse_reduce_l2>
{
    std::vector<op_desc> operators() const { return {{"ReduceL2"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto square_ins = info.add_instruction(make_op("mul"), args[0], args[0]);
        auto sum_ins    = parse_reduce_oper("reduce_sum", parser, info, {square_ins});
        return info.add_instruction(make_op("sqrt"), sum_ins);
    }
};

struct parse_reduce_log_sum : op_parser<parse_reduce_log_sum>
{
    std::vector<op_desc> operators() const { return {{"ReduceLogSum"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto sum_ins = parse_reduce_oper("reduce_sum", parser, info, std::move(args));
        return info.add_instruction(make_op("log"), sum_ins);
    }
};

struct parse_reduce_log_sum_exp : op_parser<parse_reduce_log_sum_exp>
{
    std::vector<op_desc> operators() const { return {{"ReduceLogSumExp"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        auto exp_ins = info.add_instruction(make_op("exp"), args[0]);
        auto sum_ins = parse_reduce_oper("reduce_sum", parser, info, {exp_ins});
        return info.add_instruction(make_op("log"), sum_ins);
    }
};

struct parse_reduce_sum_square : op_parser<parse_reduce_sum_square>
{
    std::vector<op_desc> operators() const { return {{"ReduceSumSquare"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        auto square_ins = info.add_instruction(make_op("mul"), args[0], args[0]);
        return parse_reduce_oper("reduce_sum", parser, std::move(info), {square_ins});
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
