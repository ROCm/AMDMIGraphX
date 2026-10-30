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
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/reshape_dims.hpp>
#include <migraphx/symbolic_shape_helpers.hpp>

#include <optional>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace onnx {

namespace {

bool has_inferred_dim(const std::vector<dim_like>& dims) { return contains(dims, dim_like{-1}); }

bool has_inferred_dim(const std::vector<sym::expr>& dims)
{
    return std::any_of(dims.begin(), dims.end(), [](const auto& dim) {
        auto value = sym::fixed_value(dim);
        return value.has_value() and sym::to<int64_t>(*value) == -1;
    });
}

} // namespace

struct parse_reshape : op_parser<parse_reshape>
{
    std::vector<op_desc> operators() const { return {{"Reshape"}}; }

    template <class T>
    static std::optional<instruction_ref> add_runtime_reshape(const onnx_parser::node_info& info,
                                                              instruction_ref input,
                                                              const std::vector<T>& reshape_dims)
    {
        const auto& input_shape = input->get_shape();
        const shape output_shape{input_shape.type(),
                                 resolve_reshape_dims(input_shape.to_symbolic(), reshape_dims)};
        validate_reshape_element_count(
            "PARSE_RESHAPE", input_shape, output_shape, has_inferred_dim(reshape_dims));
        const auto output_expressions = output_shape.sym_dims();
        auto sources = find_expression_sources(*info.mod, input, output_expressions);
        if(not sources.has_value())
            return std::nullopt;
        const auto resolved_dims = info.add_instruction(
            make_op("eval_expr_from_shape", {{"expressions", to_value(output_expressions)}}),
            *sources);
        auto allocation = info.add_instruction(
            make_op("allocate", {{"shape", to_value(output_shape)}}), resolved_dims);
        return info.add_instruction(make_op("reshape"), input, allocation);
    }

    static instruction_ref add_shape_tensor_reshape(const onnx_parser::node_info& info,
                                                    instruction_ref input,
                                                    instruction_ref shape_input)
    {
        auto allocation = info.add_instruction(
            make_op("allocate", {{"buf_type", input->get_shape().type()}}), shape_input);
        return info.add_instruction(make_op("reshape"), input, allocation);
    }

    static instruction_ref
    add_evaluated_reshape(const onnx_parser::node_info& info,
                          instruction_ref input,
                          const std::vector<int64_t>& dims,
                          std::optional<instruction_ref> shape_input = std::nullopt)
    {
        if(not input->get_shape().symbolic())
            return info.add_instruction(make_op("reshape", {{"dims", dims}}), input);
        auto result =
            add_runtime_reshape(info, input, std::vector<dim_like>{dims.begin(), dims.end()});
        if(result.has_value())
            return *result;
        if(shape_input.has_value())
            return add_shape_tensor_reshape(info, input, *shape_input);
        MIGRAPHX_THROW("PARSE_RESHAPE: symbolic dimension has no shape source");
    }

    instruction_ref parse(const op_desc& /*opd*/,
                          const onnx_parser& parser,
                          onnx_parser::node_info info,
                          std::vector<instruction_ref> args) const
    {
        std::vector<int64_t> dims;

        if(args.size() == 1)
        {
            literal s = parser.parse_value(info.attributes.at("shape"));
            s.visit([&](auto v) { copy(v, std::back_inserter(dims)); });
            return add_evaluated_reshape(info, args[0], dims);
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
                    auto result =
                        add_runtime_reshape(info, args[0], symbolic_dims.get().to_vector());
                    if(result.has_value())
                        return *result;
                }
                return add_shape_tensor_reshape(info, args[0], args[1]);
            }
            else
            {
                s.visit([&](auto v) { copy(v, std::back_inserter(dims)); });
                return add_evaluated_reshape(info, args[0], dims, args[1]);
            }
        }
    }
};

} // namespace onnx
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
