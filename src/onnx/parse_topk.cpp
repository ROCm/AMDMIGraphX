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
#include <limits>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_topk : op_parser<parse_topk>
{
    struct runtime_k_plan
    {
        int64_t axis    = 0;
        int64_t maximum = 0;
        sym::expr end;
    };

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

    static runtime_k_plan
    make_runtime_k_plan(const shape& input_shape, instruction_ref k_ins, int64_t axis)
    {
        runtime_k_plan result;
        result.axis     = tune_axis(input_shape.ndim(), axis, "TopK");
        auto axis_max   = input_shape.max_lens().at(result.axis);
        result.maximum  = axis_max > std::numeric_limits<int64_t>::max()
                              ? std::numeric_limits<int64_t>::max()
                              : static_cast<int64_t>(axis_max);
        auto symbolic_k = k_ins->sym_eval();
        if(symbolic_k.empty() or symbolic_k.get_shape().elements() != 1 or
           not shape::is_integral(symbolic_k.get_shape().type()))
            return result;

        const auto& expression = symbolic_k.get()[0];
        auto interval          = expression.eval_interval_default();
        if(not interval.valid())
            return result;
        auto min_k          = sym::to<int64_t>(interval.min);
        auto max_symbolic_k = sym::to<int64_t>(interval.max);
        if(min_k < 0 or max_symbolic_k <= 0)
            return result;
        result.maximum = std::min(result.maximum, max_symbolic_k);
        result.end     = max_symbolic_k > result.maximum
                             ? sym::resolve_min(expression, sym::lit(result.maximum))
                             : expression;
        return result;
    }

    static std::vector<instruction_ref> add_dyn_slices(const onnx_parser::node_info& info,
                                                       std::vector<instruction_ref> outputs,
                                                       instruction_ref k_ins,
                                                       int64_t axis,
                                                       const sym::expr& end)
    {
        auto starts_lit = info.add_literal(literal{{shape::int64_type, {1}}, {0}});
        auto dyn_slice =
            make_op("dyn_slice",
                    {{"axes", {axis}}, {"starts", {0}}, {"ends", value::array{to_value(end)}}});
        std::transform(outputs.begin(), outputs.end(), outputs.begin(), [&](auto output) {
            return info.add_instruction(dyn_slice, output, starts_lit, k_ins);
        });
        return outputs;
    }

    static void promote_range_outputs(const shape& input_shape,
                                      int64_t axis,
                                      const sym::expr& end,
                                      std::vector<instruction_ref>& outputs)
    {
        const auto& input_dims = input_shape.dyn_dims();
        auto axes              = range(input_dims.size());
        if(any_of(axes, [&](auto current_axis) {
               return current_axis != static_cast<std::size_t>(axis) and
                      not input_dims.at(current_axis).is_fixed();
           }))
            MIGRAPHX_THROW("PARSE_TOPK: a range-based data shape must have fixed dimensions "
                           "outside the TopK axis");

        std::vector<shape::dynamic_dimension> output_dims(input_dims.size());
        std::transform(axes.begin(), axes.end(), output_dims.begin(), [&](auto current_axis) {
            if(current_axis == static_cast<std::size_t>(axis))
                return shape::dynamic_dimension{end};
            return shape::dynamic_dimension{
                sym::lit(input_dims.at(current_axis).get_interval().min)};
        });
        std::transform(outputs.begin(), outputs.end(), outputs.begin(), [&](auto output) {
            instruction::replace(output,
                                 output->get_operator(),
                                 shape{output->get_shape().type(), output_dims},
                                 output->inputs());
            return output;
        });
    }

    static std::vector<instruction_ref>
    parse_range_runtime_k(const onnx_parser::node_info& info,
                          const std::vector<instruction_ref>& args,
                          instruction_ref k_ins,
                          const shape& input_shape,
                          const runtime_k_plan& plan,
                          bool largest)
    {
        if(plan.end.empty())
            MIGRAPHX_THROW("PARSE_TOPK: a runtime `k` needs a bounded symbolic value when the data "
                           "shape is range-based");
        auto outputs = add_topk_and_gets(info, args, plan.maximum, plan.axis, largest);
        promote_range_outputs(input_shape, plan.axis, plan.end, outputs);
        return add_dyn_slices(info, std::move(outputs), k_ins, plan.axis, plan.end);
    }

    static std::vector<instruction_ref>
    parse_symbolic_or_static_runtime_k(const onnx_parser::node_info& info,
                                       const std::vector<instruction_ref>& args,
                                       instruction_ref k_ins,
                                       const runtime_k_plan& plan,
                                       bool largest)
    {
        auto end = plan.end;
        if(end.empty())
            end = sym::var(info.name, {0, plan.maximum});
        auto outputs = add_topk_and_gets(info, args, plan.maximum, plan.axis, largest);
        return add_dyn_slices(info, std::move(outputs), k_ins, plan.axis, end);
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
        // Variable (data-dependent) `k`: run topk over the whole axis dimension, then slice the
        // outputs down to the runtime `k` using a symbolic dimension.
        const auto& input_shape = args.at(0)->get_shape();
        auto plan               = make_runtime_k_plan(input_shape, k_ins, axis);
        if(input_shape.dynamic() and not input_shape.symbolic())
            return parse_range_runtime_k(info, args, k_ins, input_shape, plan, largest);
        return parse_symbolic_or_static_runtime_k(info, args, k_ins, plan, largest);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
