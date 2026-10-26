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
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>

#include <algorithm>
#include <array>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

using substitutions         = std::unordered_map<sym::expr, sym::expr>;
using operation_replacement = std::optional<operation>;

bool is_literal(const sym::expr& e) { return e.name() == "literal"; }

// Equivalence sets for symbolic expressions. Used to simplify equivalent expressions to one
// canonical form.
struct expression_sets
{
    // disjoint-set structure for creating sets of equivalent instructions
    std::vector<sym::expr> expressions;
    std::vector<std::size_t> parents;
    // map from symbolic expression to index in `expressions`
    std::unordered_map<sym::expr, std::size_t> expression_index_map;

    std::size_t add(const sym::expr& expression)
    {
        auto [it, inserted] = expression_index_map.emplace(expression, expressions.size());
        if(inserted)
        {
            expressions.push_back(expression);
            parents.push_back(parents.size());
        }
        return it->second;
    }

    std::size_t find(std::size_t i)
    {
        if(parents[i] != i)
            parents[i] = find(parents[i]);
        return parents[i];
    }

    void merge(const sym::expr& x, const sym::expr& y)
    {
        auto x_root = find(add(x));
        auto y_root = find(add(y));
        if(x_root == y_root)
            return;
        parents[y_root] = x_root;
    }

    substitutions get_substitutions()
    {
        std::unordered_map<std::size_t, std::vector<std::size_t>> classes;
        for(std::size_t i = 0; i < expressions.size(); ++i)
            classes[find(i)].push_back(i);

        substitutions result;
        for(const auto& [root, members] : classes)
        {
            (void)root;
            auto canonical = members.front();
            auto literal   = std::find_if(
                members.begin(), members.end(), [&](auto i) { return is_literal(expressions[i]); });
            if(literal != members.end())
            {
                canonical = *literal;
                if(std::any_of(std::next(literal), members.end(), [&](auto i) {
                       return is_literal(expressions[i]) and
                              expressions[i] != expressions[canonical];
                   }))
                    MIGRAPHX_THROW("SIMPLIFY_SYMBOLIC_DIMENSIONS: conflicting symbolic literals");
            }
            for(auto i : members)
            {
                if(expressions[i] != expressions[canonical])
                    result.emplace(expressions[i], expressions[canonical]);
            }
        }
        return result;
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
    const auto* it = std::find_if(
        handlers.begin(), handlers.end(), [&](const auto& handler) { return handler.matches(op); });
    if(it == handlers.end())
        return std::nullopt;
    return it->replace(op, values);
}

} // namespace

// Unify mismatched symbolic dimensions that binary pointwise instructions require to be equal.
void simplify_symbolic_dimensions::apply(module& m) const
{
    // Collect the required dimension relationships into equivalence sets before mutating the
    // module.
    expression_sets sets;
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
                if(a.sym_expr != b.sym_expr)
                    sets.merge(a.sym_expr, b.sym_expr);
            });
    }

    // Choose one canonical expression for each equivalence set and map the other expressions to it.
    auto values = sets.get_substitutions();
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
