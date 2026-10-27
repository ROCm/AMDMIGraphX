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
#include <migraphx/simplify_symbolic_dimensions.hpp>
#include <migraphx/dim_like.hpp>
#include <migraphx/float_equal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

using substitutions         = std::unordered_map<sym::expr, sym::expr>;
using operation_replacement = std::optional<operation>;

struct runtime_extent_binding
{
    sym::expr extent;
    instruction_ref producer;
    std::vector<instruction_ref> runtime_inputs;

    static bool evaluates_to(instruction_ref input, double expected)
    {
        auto value = input->eval();
        if(value.empty() or value.get_shape().elements() != 1)
            return false;

        bool result = false;
        value.visit([&](auto v) { result = float_equal(v.front(), expected); });
        return result;
    }

    static bool is_filled_with_one(instruction_ref input)
    {
        while(input->name() == "convert" and input->inputs().size() == 1)
            input = input->inputs().front();
        if(input->name() != "fill" or input->inputs().size() != 2)
            return false;
        return evaluates_to(input->inputs().front(), 1);
    }

    std::optional<sym::expr> infer_known_extent() const
    {
        if(producer->name() != "nonzero" or producer->inputs().size() != 1)
            return std::nullopt;
        if(runtime_inputs.size() != 1)
            return std::nullopt;
        auto runtime_count = runtime_inputs.front();
        if(runtime_count->name() != "get_tuple_elem" or runtime_count->inputs().size() != 1 or
           runtime_count->inputs().front() != producer or
           runtime_count->get_operator().to_value().at("index").to<std::size_t>() != 1)
            return std::nullopt;
        auto input = producer->inputs().front();
        if(not is_filled_with_one(input))
            return std::nullopt;

        auto result      = sym::lit(std::int64_t{1});
        auto input_shape = input->get_shape().to_symbolic();
        for(const auto& d : input_shape.dyn_dims())
            result *= d.sym_expr;
        if(sym::find_variables(result).empty())
            return std::nullopt;
        return result;
    }

    bool has_same_runtime_extent(const runtime_extent_binding& other) const
    {
        if(runtime_inputs != other.runtime_inputs)
            return false;

        const bool is_range       = producer->name() == "dynamic_range";
        const bool other_is_range = other.producer->name() == "dynamic_range";
        if(is_range != other_is_range)
            return false;
        if(not is_range)
            return true;

        auto attributes       = producer->get_operator().to_value();
        auto other_attributes = other.producer->get_operator().to_value();
        return attributes.at("max_output") == other_attributes.at("max_output");
    }
};

struct runtime_extent_analysis
{
    std::unordered_map<sym::expr, runtime_extent_binding> bindings;
    substitutions known_equalities;

    static bool is_supported_producer(instruction_ref producer, std::size_t tuple_index)
    {
        if(producer->name() == "topk")
            return tuple_index < 2;
        return tuple_index == 0 and
               (producer->name() == "nonzero" or producer->name() == "nonmaxsuppression");
    }

    static bool is_expected_axis(instruction_ref producer, int64_t axis)
    {
        if(producer->name() == "nonzero")
            return axis == 1;
        if(producer->name() == "nonmaxsuppression")
            return axis == 0;
        return axis == producer->get_operator().to_value().at("axis").to<int64_t>();
    }

    static std::optional<std::pair<sym::expr, runtime_extent_binding>>
    find_slice_binding(instruction_ref ins)
    {
        if(ins->name() != "dyn_slice" or ins->inputs().size() != 3)
            return std::nullopt;

        auto attributes = ins->get_operator().to_value();
        if(not attributes.contains("axes") or not attributes.contains("starts") or
           not attributes.contains("ends") or attributes.at("axes").is_null() or
           attributes.at("starts").is_null() or attributes.at("ends").is_null())
            return std::nullopt;
        auto axes   = attributes.at("axes").to_vector<int64_t>();
        auto starts = from_value<std::vector<sym::expr>>(attributes.at("starts"));
        auto ends   = from_value<std::vector<sym::expr>>(attributes.at("ends"));
        if(axes.size() != 1 or starts.size() != 1 or ends.size() != 1 or
           ends.front().name() != "variable")
            return std::nullopt;
        auto start = sym::fixed_value(starts.front());
        if(not start.has_value() or sym::to<int64_t>(*start) != 0 or
           not runtime_extent_binding::evaluates_to(ins->inputs().at(1), 0))
            return std::nullopt;

        auto data = ins->inputs().front();
        if(data->name() != "get_tuple_elem" or data->inputs().size() != 1)
            return std::nullopt;
        auto tuple_index = data->get_operator().to_value().at("index").to<std::size_t>();
        auto producer    = data->inputs().front();
        if(not is_supported_producer(producer, tuple_index) or
           not is_expected_axis(producer, axes.front()))
            return std::nullopt;

        return std::make_pair(
            ends.front(), runtime_extent_binding{ends.front(), producer, {ins->inputs().back()}});
    }

    static std::optional<std::pair<sym::expr, runtime_extent_binding>>
    find_range_binding(instruction_ref ins)
    {
        if(ins->name() != "dynamic_range")
            return std::nullopt;
        auto attributes = ins->get_operator().to_value();
        if(not attributes.contains("output_dim") or attributes.at("output_dim").is_null())
            return std::nullopt;
        auto output_dim = from_value<shape::dynamic_dimension>(attributes.at("output_dim"));
        if(output_dim.sym_expr.name() != "variable")
            return std::nullopt;
        return std::make_pair(output_dim.sym_expr,
                              runtime_extent_binding{output_dim.sym_expr, ins, ins->inputs()});
    }

    static substitutions resolve_equalities(substitutions result)
    {
        for(std::size_t i = 0; i <= result.size(); ++i)
        {
            substitutions next;
            std::transform(result.begin(),
                           result.end(),
                           std::inserter(next, next.end()),
                           [&](const auto& item) {
                               const auto& [expression, replacement] = item;
                               return std::make_pair(expression, replacement.subs(result));
                           });
            if(next == result)
                return result;
            result = std::move(next);
        }
        MIGRAPHX_THROW("SIMPLIFY_SYMBOLIC_DIMENSIONS: cyclic runtime extent equalities");
    }

    void add_binding(sym::expr symbol, runtime_extent_binding binding)
    {
        symbol            = sym::as_symbol(symbol);
        auto known_extent = binding.infer_known_extent();
        if(known_extent.has_value() and not sym::same_symbol(symbol, *known_extent))
            known_equalities.emplace(symbol, *known_extent);

        auto same_extent = std::find_if(bindings.begin(), bindings.end(), [&](const auto& item) {
            return binding.has_same_runtime_extent(item.second);
        });
        if(same_extent != bindings.end() and same_extent->first != symbol)
            known_equalities.emplace(symbol, same_extent->second.extent);
        bindings.emplace(std::move(symbol), std::move(binding));
    }

    static runtime_extent_analysis analyze(module& m)
    {
        runtime_extent_analysis result;
        for(auto ins : iterator_for(m))
        {
            auto binding = find_slice_binding(ins);
            if(not binding.has_value())
                binding = find_range_binding(ins);
            if(binding.has_value())
                result.add_binding(std::move(binding->first), std::move(binding->second));
        }
        result.known_equalities = resolve_equalities(std::move(result.known_equalities));
        return result;
    }

    bool has_runtime_extent(const sym::expr& expression) const
    {
        auto variables = sym::find_variables(expression);
        return not variables.empty() and
               std::all_of(variables.begin(), variables.end(), [&](const auto& variable) {
                   return bindings.count(variable) > 0;
               });
    }

    sym::expr apply_known_equalities(const sym::expr& expression) const
    {
        return expression.subs(known_equalities);
    }

    void add_known_equalities(const sym::expr& expression, substitutions& result) const
    {
        for(const auto& variable : sym::find_variables(expression))
        {
            auto it = known_equalities.find(variable);
            if(it != known_equalities.end())
                result.emplace(it->first, it->second);
        }
    }

    substitutions find_pointwise_substitutions(module& m) const
    {
        substitutions result;
        for(auto ins : iterator_for(m))
        {
            const auto& inputs = ins->inputs();
            if(inputs.size() != 2 or not ins->get_operator().attributes().contains("pointwise"))
                continue;
            const auto& x = inputs.front()->get_shape();
            const auto& y = inputs.back()->get_shape();
            if(not x.symbolic() or not y.symbolic() or x.ndim() != y.ndim())
                continue;
            migraphx::for_each(
                x.dyn_dims().begin(),
                x.dyn_dims().end(),
                y.dyn_dims().begin(),
                [&](const auto& a, const auto& b) {
                    if(a.sym_expr == b.sym_expr or not has_runtime_extent(a.sym_expr) or
                       not has_runtime_extent(b.sym_expr) or
                       apply_known_equalities(a.sym_expr) != apply_known_equalities(b.sym_expr))
                        return;
                    add_known_equalities(a.sym_expr, result);
                    add_known_equalities(b.sym_expr, result);
                });
        }
        return resolve_equalities(std::move(result));
    }
};

shape substitute_shape(const shape& s, const substitutions& values)
{
    if(not s.sub_shapes().empty())
    {
        std::vector<shape> sub_shapes(s.sub_shapes().size());
        std::transform(s.sub_shapes().begin(),
                       s.sub_shapes().end(),
                       sub_shapes.begin(),
                       [&](const auto& sub) { return substitute_shape(sub, values); });
        return shape{sub_shapes};
    }
    if(not s.symbolic())
        return s;

    std::vector<shape::dynamic_dimension> dims(s.ndim());
    std::transform(s.dyn_dims().begin(), s.dyn_dims().end(), dims.begin(), [&](const auto& d) {
        return shape::dynamic_dimension{d.sym_expr.subs(values)};
    });
    std::vector<sym::expr> strides(s.ndim());
    std::transform(s.dyn_strides().begin(),
                   s.dyn_strides().end(),
                   strides.begin(),
                   [&](const auto& stride) { return stride.subs(values); });
    return shape{s.type(), std::move(dims), std::move(strides)};
}

struct dyn_slice_replacement
{
    static bool matches(const operation& op) { return op.name() == "dyn_slice"; }

    static bool substitute(value& attributes, const std::string& key, const substitutions& values)
    {
        if(not attributes.contains(key) or attributes.at(key).is_null())
            return false;
        auto expressions = from_value<std::vector<sym::expr>>(attributes.at(key));
        auto result      = expressions;
        std::transform(expressions.begin(), expressions.end(), result.begin(), [&](const auto& e) {
            return e.subs(values);
        });
        if(result == expressions)
            return false;
        attributes[key] = to_value(result);
        return true;
    }

    // Rewrite symbolic slice bounds while preserving absent optional bounds.
    static operation_replacement replace(const operation& op, const substitutions& values)
    {
        auto attributes = op.to_value();
        bool changed    = substitute(attributes, "starts", values);
        changed         = substitute(attributes, "ends", values) or changed;
        if(not changed)
            return std::nullopt;
        return make_op(op.name(), attributes);
    }
};

struct broadcast_replacement
{
    static bool matches(const operation& op)
    {
        return op.name() == "broadcast" or op.name() == "multibroadcast" or
               op.name() == "broadcast_with_dims";
    }

    // Rewrite the symbolic output dimensions shared by the broadcast operation variants.
    static operation_replacement replace(const operation& op, const substitutions& values)
    {
        auto attributes = op.to_value();
        if(not attributes.contains("out_dyn_dims") or attributes.at("out_dyn_dims").is_null())
            return std::nullopt;
        auto dims =
            from_value<std::vector<shape::dynamic_dimension>>(attributes.at("out_dyn_dims"));
        auto result = dims;
        std::transform(dims.begin(), dims.end(), result.begin(), [&](const auto& d) {
            if(not d.is_symbolic())
                return d;
            return shape::dynamic_dimension{d.sym_expr.subs(values)};
        });
        if(result == dims)
            return std::nullopt;
        attributes["out_dyn_dims"] = to_value(result);
        return make_op(op.name(), attributes);
    }
};

struct shape_operation_replacement
{
    static bool matches(const operation& op)
    {
        return op.name() == "allocate" or op.name() == "as_shape";
    }

    // Rewrite symbolic dimensions and strides stored in an operation's shape attribute.
    static operation_replacement replace(const operation& op, const substitutions& values)
    {
        auto attributes = op.to_value();
        if(not attributes.contains("shape") or attributes.at("shape").is_null())
            return std::nullopt;
        auto s      = from_value<shape>(attributes.at("shape"));
        auto result = substitute_shape(s, values);
        if(result == s)
            return std::nullopt;
        attributes["shape"] = to_value(result);
        return make_op(op.name(), attributes);
    }
};

struct dynamic_range_replacement
{
    static bool matches(const operation& op) { return op.name() == "dynamic_range"; }

    // Rewrite the optional symbolic dimension describing the range output.
    static operation_replacement replace(const operation& op, const substitutions& values)
    {
        auto attributes = op.to_value();
        if(not attributes.contains("output_dim") or attributes.at("output_dim").is_null())
            return std::nullopt;
        auto d      = from_value<shape::dynamic_dimension>(attributes.at("output_dim"));
        auto result = shape::dynamic_dimension{d.sym_expr.subs(values)};
        if(result == d)
            return std::nullopt;
        attributes["output_dim"] = to_value(result);
        return make_op(op.name(), attributes);
    }
};

struct eval_expr_from_shape_replacement
{
    static bool matches(const operation& op) { return op.name() == "eval_expr_from_shape"; }

    // Keep evaluated expressions and their captured input shapes on the same canonical symbols.
    static operation_replacement replace(const operation& op, const substitutions& values)
    {
        auto attributes = op.to_value();
        bool changed    = false;
        if(attributes.contains("expressions") and not attributes.at("expressions").is_null())
        {
            auto expressions = from_value<std::vector<sym::expr>>(attributes.at("expressions"));
            auto result      = expressions;
            std::transform(expressions.begin(),
                           expressions.end(),
                           result.begin(),
                           [&](const auto& e) { return e.subs(values); });
            if(result != expressions)
            {
                attributes["expressions"] = to_value(result);
                changed                   = true;
            }
        }
        if(attributes.contains("input_shapes") and not attributes.at("input_shapes").is_null())
        {
            auto shapes = from_value<std::vector<shape>>(attributes.at("input_shapes"));
            auto result = shapes;
            std::transform(shapes.begin(), shapes.end(), result.begin(), [&](const auto& s) {
                return substitute_shape(s, values);
            });
            if(result != shapes)
            {
                attributes["input_shapes"] = to_value(result);
                changed                    = true;
            }
        }
        if(not changed)
            return std::nullopt;
        return make_op(op.name(), attributes);
    }
};

struct reshape_replacement
{
    static bool matches(const operation& op)
    {
        return op.name() == "reshape" or op.name() == "reshape_lazy";
    }

    // Rewrite symbolic reshape dimensions while leaving concrete dimension entries unchanged.
    static operation_replacement replace(const operation& op, const substitutions& values)
    {
        auto attributes = op.to_value();
        if(not attributes.contains("dims") or attributes.at("dims").is_null())
            return std::nullopt;
        auto dims   = from_value<std::vector<dim_like>>(attributes.at("dims"));
        auto result = dims;
        std::transform(dims.begin(), dims.end(), result.begin(), [&](const auto& d) -> dim_like {
            if(not is_symbolic(d))
                return d;
            const auto& dd = std::get<shape::dynamic_dimension>(d);
            return shape::dynamic_dimension{dd.sym_expr.subs(values)};
        });
        if(result == dims)
            return std::nullopt;
        attributes["dims"] = to_value(result);
        return make_op(op.name(), attributes);
    }
};

struct replacement_handler
{
    bool (*matches)(const operation&);
    operation_replacement (*replace)(const operation&, const substitutions&);
};

// Route each supported operation to the handler that owns its complete attribute rewrite.
operation_replacement make_operation_replacement(const operation& op, const substitutions& values)
{
    static const std::array<replacement_handler, 6> handlers = {{
        {dyn_slice_replacement::matches, dyn_slice_replacement::replace},
        {broadcast_replacement::matches, broadcast_replacement::replace},
        {shape_operation_replacement::matches, shape_operation_replacement::replace},
        {dynamic_range_replacement::matches, dynamic_range_replacement::replace},
        {eval_expr_from_shape_replacement::matches, eval_expr_from_shape_replacement::replace},
        {reshape_replacement::matches, reshape_replacement::replace},
    }};
    const auto* it                                           = std::find_if(
        handlers.begin(), handlers.end(), [&](const auto& handler) { return handler.matches(op); });
    if(it == handlers.end())
        return std::nullopt;
    return it->replace(op, values);
}

} // namespace

// Unify mismatched symbolic dimensions that binary pointwise instructions require to be equal.
void simplify_symbolic_dimensions::apply(module& m) const
{
    auto extent_analysis = runtime_extent_analysis::analyze(m);
    auto values          = extent_analysis.find_pointwise_substitutions(m);
    if(values.empty())
        return;

    std::vector<instruction_ref> original;
    auto instructions = iterator_for(m);
    std::copy(instructions.begin(), instructions.end(), std::back_inserter(original));

    // Keep externally visible inputs and immutable leaves while rebuilding every computation with
    // canonical attributes and inputs.
    std::unordered_map<instruction_ref, instruction_ref> replacements;
    for(auto ins : original)
    {
        if(ins->name() == "@literal" or ins->name() == "@param" or ins->name() == "@outline" or
           ins->name() == "@comment")
            replacements.emplace(ins, ins);
    }
    auto outputs = m.add_instructions(
        original,
        &replacements,
        [&](module& mm,
            instruction_ref pos,
            const operation& op,
            const std::vector<instruction_ref>& inputs,
            const std::vector<module_ref>& module_inputs) {
            auto replacement = make_operation_replacement(op, values);
            return mm.insert_instruction(
                pos, replacement.has_value() ? *replacement : op, inputs, module_inputs);
        });
    for(const auto& [old_ins, new_ins] : replacements)
    {
        if(old_ins != new_ins)
            m.add_debug_symbols(new_ins, old_ins->get_debug_symbols());
    }
    m.replace_return(outputs);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
