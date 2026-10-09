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
#include <migraphx/onnx/symbolic_shape.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/tune_axis.hpp>

#include <algorithm>
#include <numeric>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_concat : op_parser<parse_concat>
{
    std::vector<op_desc> operators() const { return {{"Concat"}}; }

    instruction_ref parse(const op_desc& opd,
                          const onnx_parser& parser,
                          const onnx_parser::node_info& info,
                          std::vector<instruction_ref> args) const
    {
        if(args.empty())
            MIGRAPHX_THROW("PARSE_CONCAT: Concat requires at least one input");
        const auto axis_value =
            parser.parse_value(info.attributes.at("axis")).template at<int64_t>();
        const auto axis = tune_axis(args.front()->get_shape().ndim(), axis_value, opd.onnx_name);

        const auto symbolic_axis_inputs = std::count_if(args.begin(), args.end(), [&](auto arg) {
            const auto& s = arg->get_shape();
            return s.symbolic() and s.dyn_dims().at(axis).is_symbolic() and
                   s.dyn_dims().at(axis).sym_expr.name() != "literal";
        });
        if(symbolic_axis_inputs < 2)
        {
            if(any_of(args, [](auto arg) { return arg->get_shape().dynamic(); }))
                return info.add_instruction(make_op("concat", {{"axis", axis}}), args);

            args.erase(std::remove_if(args.begin(),
                                      args.end(),
                                      [](auto arg) { return arg->get_shape().elements() == 0; }),
                       args.end());
            if(args.empty())
                return info.add_instruction(make_op("undefined"));
            return info.add_instruction(make_op("concat", {{"axis", axis}}), args);
        }

        std::vector<sym::expr> axis_expressions;
        std::transform(
            args.begin(), args.end(), std::back_inserter(axis_expressions), [&](auto arg) {
                return arg->get_shape().to_symbolic().dyn_dims().at(axis).sym_expr;
            });

        std::vector<instruction_ref> counts;
        std::transform(
            args.begin(),
            args.end(),
            axis_expressions.begin(),
            std::back_inserter(counts),
            [&](auto arg, const auto& expression) {
                auto sources = find_expression_sources(*info.mod, arg, {expression});
                if(not sources.has_value())
                    MIGRAPHX_THROW("PARSE_CONCAT: cannot resolve symbolic input extent");
                return info.add_instruction(
                    make_op("eval_expr_from_shape",
                            {{"expressions", to_value(std::vector<sym::expr>{expression})}}),
                    *sources);
            });

        auto dyn_concat_inputs = args;
        dyn_concat_inputs.insert(dyn_concat_inputs.end(), counts.begin(), counts.end());
        auto concat =
            info.add_instruction(make_op("dyn_concat", {{"axis", axis}}), dyn_concat_inputs);
        auto buffer = info.add_instruction(make_op("get_tuple_elem", {{"index", 0}}), concat);
        auto total  = info.add_instruction(make_op("get_tuple_elem", {{"index", 1}}), concat);

        const auto total_expression = std::accumulate(axis_expressions.begin(),
                                                      axis_expressions.end(),
                                                      sym::lit(int64_t{0}),
                                                      std::plus<sym::expr>{});
        const auto total_interval   = shape::dynamic_dimension{total_expression}.get_interval();
        const auto output_dim       = sym::var(info.name, {total_interval.min, total_interval.max});
        auto starts                 = info.add_literal(literal{{shape::int64_type, {1}}, {0}});
        return info.add_instruction(make_op("dyn_slice",
                                            {{"axes", {axis}},
                                             {"starts", {0}},
                                             {"ends", value::array{to_value(output_dim)}},
                                             {"always_leq", true}}),
                                    buffer,
                                    starts,
                                    total);
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
