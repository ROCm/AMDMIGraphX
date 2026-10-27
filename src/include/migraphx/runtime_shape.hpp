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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_RUNTIME_SHAPE_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_RUNTIME_SHAPE_HPP

#include <migraphx/dim_like.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/reshape_dims.hpp>
#include <migraphx/value.hpp>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

inline std::size_t fixed_dimension(const shape& s,
                                   std::size_t axis,
                                   const std::string& operation_name,
                                   const std::string& dimension_name)
{
    if(not s.dynamic())
        return s.lens().at(axis);
    auto result = sym::fixed_value(s.dyn_dims().at(axis).sym_expr);
    if(not result.has_value())
        MIGRAPHX_THROW(operation_name + ": " + dimension_name + " must be fixed");
    return sym::to<std::size_t>(*result);
}

namespace runtime_shape_detail {

inline bool shape_source_has_symbol(instruction_ref source, const sym::expr& symbol)
{
    const auto& source_shape = source->get_shape();
    return source_shape.symbolic() and any_of(source_shape.dyn_dims(), [&](const auto& d) {
               return d.sym_expr.name() == "variable" and sym::as_symbol(d.sym_expr) == symbol;
           });
}

inline bool shape_source_has_expression(instruction_ref source, const sym::expr& expression)
{
    const auto& source_shape = source->get_shape();
    return source_shape.symbolic() and
           any_of(source_shape.dyn_dims(), [&](const auto& d) { return d.sym_expr == expression; });
}

inline bool can_evaluate_shape_expressions(const std::vector<instruction_ref>& shape_sources,
                                           const std::vector<sym::expr>& expressions)
{
    return not shape_sources.empty() and all_of(expressions, [&](const auto& expression) {
        if(any_of(shape_sources,
                  [&](auto source) { return shape_source_has_expression(source, expression); }))
            return true;
        return all_of(sym::find_variables(expression), [&](const auto& symbol) {
            return any_of(shape_sources,
                          [&](auto source) { return shape_source_has_symbol(source, symbol); });
        });
    });
}

inline std::vector<shape::dynamic_dimension>
resolve_runtime_reshape_dims(const shape& input, const std::vector<dim_like>& dims)
{
    return resolve_reshape_dims(input, dims);
}

inline std::vector<shape::dynamic_dimension>
resolve_runtime_reshape_dims(const shape& input, const std::vector<sym::expr>& dims)
{
    return resolve_reshape_dims(input, dims);
}

inline std::vector<shape::dynamic_dimension>
resolve_runtime_reshape_dims(const shape&, const std::vector<shape::dynamic_dimension>& dims)
{
    return dims;
}

template <class Inserter>
std::optional<instruction_ref>
try_runtime_allocation(shape::type_t type,
                       const std::vector<shape::dynamic_dimension>& dims,
                       const std::vector<instruction_ref>& shape_sources,
                       Inserter insert)
{
    std::vector<sym::expr> expressions(dims.size());
    transform(dims, expressions.begin(), [](const auto& dim) { return dim.sym_expr; });
    if(not can_evaluate_shape_expressions(shape_sources, expressions))
        return std::nullopt;
    auto resolved_dims = insert(
        make_op("eval_expr_from_shape", {{"expressions", to_value(expressions)}}), shape_sources);
    return insert(make_op("allocate", {{"shape", to_value(shape{type, dims})}}),
                  std::vector<instruction_ref>{resolved_dims});
}

template <class T, class Inserter>
std::optional<instruction_ref>
try_runtime_reshape(instruction_ref input,
                    const std::vector<T>& reshape_dims,
                    const std::vector<instruction_ref>& shape_sources,
                    Inserter insert)
{
    const auto output_dims =
        resolve_runtime_reshape_dims(input->get_shape().to_symbolic(), reshape_dims);
    auto allocation =
        try_runtime_allocation(input->get_shape().type(), output_dims, shape_sources, insert);
    if(not allocation.has_value())
        return std::nullopt;
    return insert(make_op("reshape"), std::vector<instruction_ref>{input, *allocation});
}

} // namespace runtime_shape_detail

inline std::optional<instruction_ref>
try_add_runtime_allocation(module& m,
                           shape::type_t type,
                           const std::vector<shape::dynamic_dimension>& dims,
                           const std::vector<instruction_ref>& shape_sources)
{
    return runtime_shape_detail::try_runtime_allocation(
        type, dims, shape_sources, [&](const auto& op, const auto& args) {
            return m.add_instruction(op, args);
        });
}

template <class T>
std::optional<instruction_ref>
try_add_runtime_reshape(module& m,
                        instruction_ref input,
                        const std::vector<T>& dims,
                        const std::vector<instruction_ref>& shape_sources)
{
    return runtime_shape_detail::try_runtime_reshape(
        input, dims, shape_sources, [&](const auto& op, const auto& args) {
            return m.add_instruction(op, args);
        });
}

template <class T>
std::optional<instruction_ref>
try_insert_runtime_reshape(module& m,
                           instruction_ref ins,
                           instruction_ref input,
                           const std::vector<T>& dims,
                           const std::vector<instruction_ref>& shape_sources)
{
    return runtime_shape_detail::try_runtime_reshape(
        input, dims, shape_sources, [&](const auto& op, const auto& args) {
            return m.insert_instruction(ins, op, args);
        });
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
