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
#include <migraphx/onnx/symbolic_shape.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/reshape_dims.hpp>

#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

struct parse_reshape : op_parser<parse_reshape>
{
    std::vector<op_desc> operators() const { return {{"Reshape"}}; }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        std::vector<int64_t> dims;
        auto add_runtime_reshape = [&](const auto& reshape_dims) {
            const auto& input_shape = args[0]->get_shape();
            const auto output_dims  = resolve_reshape_dims(input_shape.to_symbolic(), reshape_dims);
            std::vector<sym::expr> output_expressions(output_dims.size());
            transform(output_dims, output_expressions.begin(), [](const auto& dim) {
                return dim.sym_expr;
            });
            auto sources = find_expression_sources(*info.mod, args[0], output_expressions);
            if(not sources.has_value())
                return std::optional<instruction_ref>{};
            const auto resolved_dims = info.add_instruction(
                make_op("eval_expr_from_shape", {{"expressions", to_value(output_expressions)}}),
                *sources);
            const shape output_shape{input_shape.type(), output_dims};
            auto allocation = info.add_instruction(
                make_op("allocate", {{"shape", to_value(output_shape)}}), resolved_dims);
            return std::optional<instruction_ref>{
                info.add_instruction(make_op("reshape"), args[0], allocation)};
        };
        auto add_shape_tensor_reshape = [&] {
            assert(args.size() == 2);
            const auto& input_shape = args[0]->get_shape();
            auto allocation         = info.add_instruction(
                make_op("allocate", {{"buf_type", input_shape.type()}}), args[1]);
            return info.add_instruction(make_op("reshape"), args[0], allocation);
        };
        auto add_evaluated_reshape = [&] {
            if(not args[0]->get_shape().symbolic())
                return info.add_instruction(make_op("reshape", {{"dims", dims}}), args[0]);
            auto result = add_runtime_reshape(std::vector<dim_like>{dims.begin(), dims.end()});
            if(not result.has_value())
            {
                if(args.size() == 2)
                    return add_shape_tensor_reshape();
                MIGRAPHX_THROW("PARSE_RESHAPE: symbolic dimension has no direct shape source");
            }
            return *result;
        };

        if(args.size() == 1)
        {
            literal s = parser.parse_value(info.attributes.at("shape"));
            s.visit([&](auto v) { copy(v, std::back_inserter(dims)); });
            return add_evaluated_reshape();
        }
        else
        {
            // 2 inputs
            auto s = args[1]->eval();
            if(s.empty())
            {
                // arg[1] not eval-able
                const auto symbolic_dims = args[1]->sym_eval();
                const auto& input_shape  = args[0]->get_shape();
                if(not symbolic_dims.empty() and
                   (not input_shape.dynamic() or input_shape.symbolic()))
                {
                    auto result = add_runtime_reshape(symbolic_dims.get().to_vector());
                    if(result.has_value())
                        return *result;
                }
                return add_shape_tensor_reshape();
            }
            else
            {
                s.visit([&](auto v) { copy(v, std::back_inserter(dims)); });
                return add_evaluated_reshape();
            }
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
