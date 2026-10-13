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
#include <migraphx/module.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/sym.hpp>

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

using substitutions = std::unordered_map<sym::expr, sym::expr>;

bool is_literal(const sym::expr& e) { return e.name() == "literal"; }

struct expression_sets
{
    std::unordered_map<sym::expr, std::size_t> indices;
    std::vector<sym::expr> expressions;
    std::vector<std::size_t> parents;

    std::size_t add(const sym::expr& expression)
    {
        auto [it, inserted] = indices.emplace(expression, expressions.size());
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

bool substitute_exprs(value& attributes, const std::string& key, const substitutions& values)
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

bool substitute_dims(value& attributes, const std::string& key, const substitutions& values)
{
    if(not attributes.contains(key) or attributes.at(key).is_null())
        return false;
    auto dims   = from_value<std::vector<shape::dynamic_dimension>>(attributes.at(key));
    auto result = dims;
    std::transform(dims.begin(), dims.end(), result.begin(), [&](const auto& d) {
        return shape::dynamic_dimension{d.sym_expr.subs(values)};
    });
    if(result == dims)
        return false;
    attributes[key] = to_value(result);
    return true;
}

bool substitute_shapes(value& attributes, const std::string& key, const substitutions& values)
{
    if(not attributes.contains(key) or attributes.at(key).is_null())
        return false;
    auto shapes = from_value<std::vector<shape>>(attributes.at(key));
    auto result = shapes;
    std::transform(shapes.begin(), shapes.end(), result.begin(), [&](const auto& s) {
        return substitute_shape(s, values);
    });
    if(result == shapes)
        return false;
    attributes[key] = to_value(result);
    return true;
}

bool substitute_single_shape(value& attributes, const std::string& key, const substitutions& values)
{
    if(not attributes.contains(key) or attributes.at(key).is_null())
        return false;
    auto s      = from_value<shape>(attributes.at(key));
    auto result = substitute_shape(s, values);
    if(result == s)
        return false;
    attributes[key] = to_value(result);
    return true;
}

bool substitute_optional_dim(value& attributes, const std::string& key, const substitutions& values)
{
    if(not attributes.contains(key) or attributes.at(key).is_null())
        return false;
    auto d      = from_value<shape::dynamic_dimension>(attributes.at(key));
    auto result = shape::dynamic_dimension{d.sym_expr.subs(values)};
    if(result == d)
        return false;
    attributes[key] = to_value(result);
    return true;
}

bool substitute_dim_likes(value& attributes, const std::string& key, const substitutions& values)
{
    if(not attributes.contains(key) or attributes.at(key).is_null())
        return false;
    auto dims   = from_value<std::vector<dim_like>>(attributes.at(key));
    auto result = dims;
    std::transform(dims.begin(), dims.end(), result.begin(), [&](const auto& d) -> dim_like {
        if(not is_symbolic(d))
            return d;
        const auto& dd = std::get<shape::dynamic_dimension>(d);
        return shape::dynamic_dimension{dd.sym_expr.subs(values)};
    });
    if(result == dims)
        return false;
    attributes[key] = to_value(result);
    return true;
}

operation substitute_operation(const operation& op, const substitutions& values)
{
    auto attributes = op.to_value();
    bool changed    = false;
    if(op.name() == "dyn_slice")
    {
        changed = substitute_exprs(attributes, "starts", values) or changed;
        changed = substitute_exprs(attributes, "ends", values) or changed;
    }
    if(contains({"broadcast", "multibroadcast", "broadcast_with_dims"}, op.name()))
        changed = substitute_dims(attributes, "out_dyn_dims", values) or changed;
    if(op.name() == "allocate" or op.name() == "as_shape")
        changed = substitute_single_shape(attributes, "shape", values) or changed;
    if(op.name() == "dynamic_range")
        changed = substitute_optional_dim(attributes, "output_dim", values) or changed;
    if(op.name() == "eval_expr_from_shape")
    {
        changed = substitute_exprs(attributes, "expressions", values) or changed;
        changed = substitute_shapes(attributes, "input_shapes", values) or changed;
    }
    if(op.name() == "reshape" or op.name() == "reshape_lazy")
        changed = substitute_dim_likes(attributes, "dims", values) or changed;
    if(not changed)
        return op;
    auto result = op;
    result.from_value(attributes);
    return result;
}

std::vector<instruction_ref>
replace_inputs(const std::vector<instruction_ref>& inputs,
               const std::unordered_map<instruction_ref, instruction_ref>& replacements)
{
    std::vector<instruction_ref> result(inputs.size());
    std::transform(inputs.begin(), inputs.end(), result.begin(), [&](auto input) {
        auto it = replacements.find(input);
        return it == replacements.end() ? input : it->second;
    });
    return result;
}

} // namespace

void simplify_symbolic_dimensions::apply(module& m) const
{
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
            x.dyn_dims().begin(), x.dyn_dims().end(), y.dyn_dims().begin(), [&](auto a, auto b) {
                if(a.sym_expr != b.sym_expr)
                    sets.merge(a.sym_expr, b.sym_expr);
            });
    }

    auto values = sets.get_substitutions();
    if(values.empty())
        return;

    std::vector<instruction_ref> original;
    auto instructions = iterator_for(m);
    std::copy(instructions.begin(), instructions.end(), std::back_inserter(original));
    std::unordered_map<instruction_ref, instruction_ref> replacements;
    std::vector<instruction_ref> old_parameters;
    instruction_ref return_ins = m.end();

    for(auto ins : original)
    {
        if(ins->name() == "@literal" or ins->name() == "@outline" or ins->name() == "@comment")
        {
            replacements.emplace(ins, ins);
            continue;
        }
        if(ins->name() == "@return")
        {
            return_ins = ins;
            continue;
        }
        if(ins->name() == "@param")
        {
            auto new_shape = substitute_shape(ins->get_shape(), values);
            if(new_shape == ins->get_shape())
            {
                replacements.emplace(ins, ins);
                continue;
            }
            auto name = ins->get_operator().to_value().at("parameter").to<std::string>();
            m.rename_parameter(ins, name + "#simplify_symbolic_dimensions_old");
            auto replacement = m.insert_parameter(ins, name, new_shape);
            replacements.emplace(ins, replacement);
            old_parameters.push_back(ins);
            continue;
        }

        auto args        = replace_inputs(ins->inputs(), replacements);
        auto replacement = m.insert_instruction(
            ins, substitute_operation(ins->get_operator(), values), args, ins->module_inputs());
        m.add_debug_symbols(replacement, ins->get_debug_symbols());
        replacements.emplace(ins, replacement);
    }

    if(return_ins != m.end())
        m.replace_return(replace_inputs(return_ins->inputs(), replacements));

    for(auto it = original.rbegin(); it != original.rend(); ++it)
    {
        auto ins = *it;
        if(ins->name() == "@return" or ins->name() == "@literal" or ins->name() == "@outline" or
           ins->name() == "@comment" or ins->name() == "@param")
            continue;
        m.remove_instruction(ins);
    }
    for(auto ins : old_parameters)
        m.remove_instruction(ins);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
