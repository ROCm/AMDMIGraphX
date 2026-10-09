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

#include <migraphx/split_sym_dim.hpp>
#include <migraphx/split_sym/analyzer.hpp>
#include <migraphx/split_sym/slice.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/eliminate_convert.hpp>
#include <migraphx/eliminate_identity.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/module.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/sym.hpp>
#include <migraphx/value.hpp>
#include <migraphx/zip_view.hpp>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using namespace migraphx::split_sym; // NOLINT

namespace {

struct sliced_value
{
    instruction_ref source;
    std::vector<std::size_t> slice_axes;
};

bool operator==(const sliced_value& x, const sliced_value& y)
{
    return x.source == y.source and x.slice_axes == y.slice_axes;
}

struct clone_input_plan
{
    sliced_value clone_value;
    operand_plan operand;
};

struct symbolic_op_plan
{
    symbolic_op_info info;
    bool root_source = false;
    std::size_t rank = 0;
    std::optional<std::size_t> block;
    std::vector<clone_input_plan> clone_inputs;
    shape dispatch_output;
};

struct root_registry
{
    std::unordered_map<sym::expr, instruction_ref> sources;
    std::unordered_set<sym::expr> data_roots;
    std::unordered_set<instruction_ref> boundaries;
    std::unordered_map<instruction_ref, instruction_ref> dyn_concat_boundaries;
};

std::optional<instruction_ref> find_dyn_concat(instruction_ref slice)
{
    if(slice->name() != "dyn_slice" or slice->inputs().size() != 3)
        return std::nullopt;
    auto buffer = slice->inputs().front();
    auto count  = slice->inputs().back();
    if(buffer->name() != "get_tuple_elem" or count->name() != "get_tuple_elem" or
       buffer->inputs().size() != 1 or count->inputs().size() != 1)
        return std::nullopt;
    auto buffer_attributes = buffer->get_operator().to_value();
    auto count_attributes  = count->get_operator().to_value();
    if(buffer_attributes.at("index").to<std::size_t>() != 0 or
       count_attributes.at("index").to<std::size_t>() != 1)
        return std::nullopt;
    auto concat = buffer->inputs().front();
    if(concat != count->inputs().front() or concat->name() != "dyn_concat")
        return std::nullopt;
    return concat;
}

bool register_shape_roots(root_registry& registry, instruction_ref source)
{
    const auto& s = source->get_shape();
    if(not s.symbolic())
        return false;
    bool inserted = false;
    for(const auto& d : s.dyn_dims())
    {
        if(not d.is_symbolic() or d.sym_expr.name() != "variable")
            continue;
        inserted = registry.sources.emplace(sym::as_symbol(d.sym_expr), source).second or inserted;
    }
    return inserted;
}

root_registry find_root_sources(const module& m)
{
    root_registry result;
    for(const auto& name : m.get_parameter_names())
        register_shape_roots(result, m.get_parameter(name));
    for(auto ins : iterator_for(m))
    {
        if(ins->name() != "dyn_slice" or not ins->get_shape().symbolic())
            continue;
        bool boundary = false;
        for(const auto& d : ins->get_shape().dyn_dims())
        {
            if(not d.is_symbolic() or d.sym_expr.name() != "variable")
                continue;
            auto root = sym::as_symbol(d.sym_expr);
            if(result.sources.emplace(root, ins).second)
                result.data_roots.insert(root);
            boundary = boundary or contains(result.data_roots, root);
        }
        if(boundary)
        {
            result.boundaries.insert(ins);
            auto dyn_concat = find_dyn_concat(ins);
            if(dyn_concat.has_value())
                result.dyn_concat_boundaries.emplace(ins, *dyn_concat);
        }
    }
    return result;
}

std::optional<std::vector<instruction_ref>>
find_expression_sources(const std::vector<sym::expr>& expressions,
                        const std::unordered_map<sym::expr, instruction_ref>& root_sources)
{
    std::unordered_set<sym::expr> required;
    for(const auto& expression : expressions)
    {
        auto variables = sym::find_variables(expression);
        required.insert(variables.begin(), variables.end());
    }
    std::vector<std::pair<std::string, instruction_ref>> ordered_sources;
    std::unordered_set<instruction_ref> seen;
    for(const auto& variable : required)
    {
        auto source = root_sources.find(variable);
        if(source == root_sources.end())
            return std::nullopt;
        if(seen.insert(source->second).second)
            ordered_sources.emplace_back(variable.to_string(), source->second);
    }
    std::sort(ordered_sources.begin(), ordered_sources.end(), [](const auto& x, const auto& y) {
        return x.first < y.first;
    });
    std::vector<instruction_ref> result;
    std::transform(ordered_sources.begin(),
                   ordered_sources.end(),
                   std::back_inserter(result),
                   [](const auto& entry) { return entry.second; });
    return result;
}

struct resolve_symbolic_dimensions_of_match : match::supports_dynamic_shapes
{
    std::unordered_map<sym::expr, instruction_ref> root_sources;

    auto matcher() const { return match::name("dimensions_of")(match::nargs(1)); }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins                = mr.result;
        const auto& input_shape = ins->inputs().front()->get_shape();
        if(not input_shape.symbolic())
            return;
        const auto symbolic_value = ins->sym_eval();
        if(symbolic_value.empty())
            return;
        const auto expressions = symbolic_value.get().to_vector();
        auto sources           = find_expression_sources(expressions, root_sources);
        if(not sources.has_value())
            return;
        if(sources->empty())
        {
            std::vector<int64_t> values;
            std::transform(expressions.begin(),
                           expressions.end(),
                           std::back_inserter(values),
                           [](const auto& expression) {
                               return static_cast<int64_t>(expression.eval_uint({}));
                           });
            m.replace_instruction(
                ins, m.add_literal(literal{shape{shape::int64_type, {values.size()}}, values}));
        }
        else
        {
            m.replace_instruction(
                ins,
                make_op("eval_expr_from_shape", {{"expressions", to_value(expressions)}}),
                *sources);
        }
    }
};

struct find_eval_expr_from_intermediate_ins : match::supports_dynamic_shapes
{
    std::unordered_map<sym::expr, instruction_ref> root_sources;

    auto matcher() const
    {
        auto non_parameter           = match::none_of(match::name("@param"));
        auto has_non_parameter_input = match::any_of[match::inputs()](non_parameter);
        return match::name("eval_expr_from_shape")(has_non_parameter_input);
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins = mr.result;
        auto expressions =
            from_value<std::vector<sym::expr>>(ins->get_operator().to_value().at("expressions"));
        auto expression_sources = find_expression_sources(expressions, root_sources);
        if(not expression_sources.has_value() or expression_sources->empty() or
           any_of(*expression_sources,
                  [](instruction_ref source) { return source->name() != "@param"; }))
            return;
        m.replace_instruction(
            ins,
            make_op("eval_expr_from_shape", {{"expressions", to_value(expressions)}}),
            *expression_sources);
    }
};

bool zeros_contracted_region(const axis_mask& source, const axis_mask& consumer)
{
    return source.role == mask_role::normalized and source.fill == fill_kind::neg_inf and
           consumer.role == mask_role::contracted and consumer.fill == fill_kind::zero and
           source.axis == consumer.axis and sym::same_symbol(source.extent, consumer.extent);
}

void remove_redundant_masks(
    std::vector<symbolic_op_plan>& infos,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    for(auto& plan : infos)
    {
        auto& info       = plan.info;
        const auto& args = info.ins->inputs();
        for(auto&& [arg, operand] : views::zip(args, info.operands))
        {
            auto source = info_for_instruction.find(arg);
            if(source == info_for_instruction.end())
                continue;
            auto& masks = operand.masks;
            masks.erase(std::remove_if(
                            masks.begin(),
                            masks.end(),
                            [&](const auto& mask) {
                                return any_of(
                                    source->second->info.operands, [&](const auto& source_operand) {
                                        return any_of(
                                            source_operand.masks, [&](const auto& source_mask) {
                                                return zeros_contracted_region(source_mask, mask);
                                            });
                                    });
                            }),
                        masks.end());
        }
    }
}

struct root_spec
{
    sym::expr root;
    std::string name;
    shape::dynamic_dimension::interval interval;
    sym::expr target_symbol;
    std::map<std::size_t, shape::dynamic_dimension::interval> specializations;
};

bool collect_root_dimension(const shape::dynamic_dimension& dimension,
                            std::unordered_map<sym::expr, root_spec>& root_specs)
{
    if(not dimension.is_symbolic())
        return true;
    if(dimension.sym_expr.name() != "variable")
        return not is_variable_axis(dimension);

    auto root        = sym::as_symbol(dimension.sym_expr);
    auto name        = root.to_string();
    auto interval    = dimension.get_interval();
    auto optimal_set = dimension.get_optimals();
    if(any_of(optimal_set, [&](auto x) { return x < interval.min or x > interval.max; }))
        return false;
    optimal_set.insert(interval.min);
    optimal_set.insert(interval.max);
    std::map<std::size_t, shape::dynamic_dimension::interval> specializations;
    auto lower = interval.min;
    for(auto optimal : optimal_set)
    {
        specializations.emplace(optimal, shape::dynamic_dimension::interval{lower, optimal});
        lower = optimal + 1;
    }

    auto found = root_specs.find(root);
    if(found != root_specs.end())
        return found->second.interval == interval and
               found->second.specializations == specializations;

    root_specs.emplace(root,
                       root_spec{root, std::move(name), interval, {}, std::move(specializations)});
    return true;
}

// Collect the independent symbols that enter through module parameters. Each symbol's min,
// optimals, and max partition its runtime interval into specialization buckets.
std::optional<std::vector<root_spec>> collect_roots(const module& m, const root_registry& registry)
{
    std::unordered_map<sym::expr, root_spec> root_specs;
    auto parameter_names = m.get_parameter_names();
    for(const auto& parameter_name : parameter_names)
    {
        const auto& s = m.get_parameter(parameter_name)->get_shape();
        if(s.dynamic() and not s.symbolic())
            return std::nullopt;
        if(not s.symbolic())
            continue;
        if(not all_of(s.dyn_dims(), [&](const auto& dimension) {
               return collect_root_dimension(dimension, root_specs);
           }))
            return std::nullopt;
    }
    for(const auto& [root, source] : registry.sources)
    {
        if(source->name() == "@param")
            continue;
        bool found = false;
        for(const auto& dimension : source->get_shape().dyn_dims())
        {
            if(not dimension.is_symbolic() or dimension.sym_expr.name() != "variable" or
               sym::as_symbol(dimension.sym_expr) != root)
                continue;
            found = true;
            if(not collect_root_dimension(dimension, root_specs))
                return std::nullopt;
        }
        if(not found)
            return std::nullopt;
    }
    if(root_specs.empty())
        return std::nullopt;

    std::vector<root_spec> roots;
    roots.reserve(root_specs.size());
    std::transform(root_specs.begin(),
                   root_specs.end(),
                   std::back_inserter(roots),
                   [](auto& entry) { return std::move(entry.second); });
    std::sort(
        roots.begin(), roots.end(), [](const auto& x, const auto& y) { return x.name < y.name; });
    std::unordered_set<std::string> symbol_names;
    for(const auto& root : roots)
        symbol_names.insert(root.name);

    for(auto& root : roots)
    {
        std::string target_name = "split_sym_dim_" + root.name + "_target";
        while(contains(symbol_names, target_name))
            target_name += "_";
        symbol_names.insert(target_name);
        std::set<sym::scalar> targets;
        for(const auto& specialization : root.specializations)
            targets.insert(sym::scalar{specialization.first});
        if(root.interval.min == root.interval.max)
            root.target_symbol = sym::lit(root.interval.min);
        else
            root.target_symbol = sym::var(
                std::move(target_name), {root.interval.min, root.interval.max}, std::move(targets));
    }
    return roots;
}

struct block_plan
{
    std::vector<symbolic_op_plan*> ops;
    std::vector<const root_spec*> roots;
};

std::unordered_set<sym::expr> get_block_data_roots(const block_plan& block,
                                                   const std::unordered_set<sym::expr>& data_roots)
{
    std::unordered_set<sym::expr> result;
    for(const auto* root : block.roots)
        if(contains(data_roots, root->root))
            result.insert(root->root);
    return result;
}

bool same_block_roots(const block_plan& x, const block_plan& y)
{
    return x.roots.size() == y.roots.size() and
           std::equal(x.roots.begin(), x.roots.end(), y.roots.begin(), [](auto* a, auto* b) {
               return a->root == b->root;
           });
}

shape substitute_shape(const shape& s,
                       const std::unordered_map<sym::expr, sym::expr>& substitutions)
{
    if(not s.symbolic())
        return s;
    std::vector<shape::dynamic_dimension> dimensions(s.ndim());
    std::transform(
        s.dyn_dims().begin(), s.dyn_dims().end(), dimensions.begin(), [&](const auto& d) {
            return shape::dynamic_dimension{d.sym_expr.subs(substitutions)};
        });
    std::vector<sym::expr> strides(s.ndim());
    std::transform(s.dyn_strides().begin(),
                   s.dyn_strides().end(),
                   strides.begin(),
                   [&](const auto& stride) { return stride.subs(substitutions); });
    return {s.type(), dimensions, strides};
}

void gather_shape_roots(const shape& s, std::unordered_set<sym::expr>& result)
{
    if(not s.symbolic())
        return;
    for(const auto& d : s.dyn_dims())
    {
        auto variables = sym::find_variables(d.sym_expr);
        result.insert(variables.begin(), variables.end());
    }
    for(const auto& stride : s.dyn_strides())
    {
        auto variables = sym::find_variables(stride);
        result.insert(variables.begin(), variables.end());
    }
}

bool roots_fit_clone_limit(const std::vector<const root_spec*>& roots, std::size_t max_clones)
{
    std::size_t clone_count = 1;
    for(const auto* root : roots)
    {
        if(root->specializations.size() > std::numeric_limits<std::size_t>::max() / clone_count)
            return false;
        clone_count *= root->specializations.size();
        if(max_clones != 0 and clone_count > max_clones)
            return false;
    }
    return true;
}

std::unordered_set<sym::expr> find_info_roots(const symbolic_op_info& info)
{
    std::unordered_set<sym::expr> required;
    gather_shape_roots(info.get_output_shape(), required);
    for(const auto& input : info.get_input_shapes())
        gather_shape_roots(input, required);
    for(const auto& operand : info.operands)
        for(const auto& mask : operand.masks)
        {
            auto variables = sym::find_variables(mask.extent);
            required.insert(variables.begin(), variables.end());
        }
    return required;
}

std::optional<std::vector<const root_spec*>> find_instruction_roots(
    const symbolic_op_info& info, const std::vector<root_spec>& roots, std::size_t max_clones)
{
    auto required = find_info_roots(info);
    std::vector<const root_spec*> result;
    for(const auto& root : roots)
    {
        if(not contains(required, root.root))
            continue;
        required.erase(root.root);
        result.push_back(&root);
    }
    if(not required.empty() or result.empty() or not roots_fit_clone_limit(result, max_clones))
        return std::nullopt;
    return result;
}

std::unordered_set<sym::expr> get_owned_roots(const block_plan& block)
{
    std::unordered_set<sym::expr> result;
    std::transform(block.roots.begin(),
                   block.roots.end(),
                   std::inserter(result, result.end()),
                   [](const auto* root) { return root->root; });
    return result;
}

bool freezer_roots_are_owned(
    instruction_ref ins,
    const std::unordered_set<sym::expr>& owned_roots,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    auto found = info_for_instruction.find(ins);
    if(found == info_for_instruction.end() or not found->second->info.freezer or
       ins->name() != "reshape" or ins->inputs().size() != 2)
        return true;
    auto required = find_info_roots(found->second->info);
    return all_of(required, [&](const auto& root) { return contains(owned_roots, root); });
}

bool owned_shape_transform_dependency(
    instruction_ref ins,
    const std::unordered_set<sym::expr>& owned_roots,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    if(not contains({"contiguous", "flatten", "reshape", "transpose"}, ins->name()))
        return false;
    auto found = info_for_instruction.find(ins);
    if(found == info_for_instruction.end())
        return false;
    const auto& info                       = found->second->info;
    auto preserves_symbolic_axis_positions = [&] {
        auto input_shapes = info.get_input_shapes();
        if(input_shapes.empty())
            return false;
        const auto& input  = input_shapes.front();
        const auto& output = info.get_output_shape();
        if(not input.symbolic() or not output.symbolic())
            return false;
        auto find_variable_axes = [](const shape& s) {
            std::unordered_map<sym::expr, std::size_t> result;
            for(std::size_t axis = 0; axis < s.ndim(); ++axis)
                for(const auto& variable : sym::find_variables(s.dyn_dims().at(axis).sym_expr))
                    if(not result.emplace(variable, axis).second)
                        return std::optional<decltype(result)>{};
            return std::optional<decltype(result)>{std::move(result)};
        };
        auto input_axes  = find_variable_axes(input);
        auto output_axes = find_variable_axes(output);
        return input_axes.has_value() and output_axes.has_value() and *input_axes == *output_axes;
    };
    bool freezable_reshape = ins->name() == "reshape" and ins->inputs().size() == 2 and
                             static_cast<bool>(info.freezer) and
                             preserves_symbolic_axis_positions();
    if(not info.supported and not freezable_reshape)
        return false;
    auto required = find_info_roots(info);
    return all_of(required, [&](const auto& root) { return contains(owned_roots, root); });
}

bool can_specialize(const symbolic_op_plan& plan)
{
    const auto& info = plan.info;
    const bool needs_padding =
        any_of(info.operands, [](const auto& operand) { return operand.pad_value.has_value(); });
    const bool needs_specialization = info.freezer or needs_padding;
    return not plan.root_source and not info.output_symbolic_axes.empty() and info.supported and
           info.ins->module_inputs().empty() and needs_specialization;
}

bool absorbable_dependency(
    instruction_ref ins,
    const std::unordered_set<instruction_ref>& planned,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    if(contains(planned, ins))
        return true;
    auto found = info_for_instruction.find(ins);
    if(found != info_for_instruction.end() and found->second->root_source)
        return false;
    if(starts_with(ins->name(), "@") or not ins->module_inputs().empty())
        return false;
    if(ins->get_operator().attributes().contains("reduce"))
    {
        const auto& inputs = ins->inputs();
        if(any_of(inputs, [&](instruction_ref input) { return contains(planned, input); }))
            return false;
    }
    const auto& s = ins->get_shape();
    if(not s.dynamic())
        return true;
    if(not s.symbolic())
        return false;
    if(found != info_for_instruction.end() and ins->name() == "reshape" and
       ins->inputs().size() == 2 and not found->second->info.supported)
        return false;
    if(found != info_for_instruction.end() and
       contains({"gather", "scatter_none", "step", "concat"}, ins->name()) and
       not found->second->info.supported)
        return false;
    return all_of(s.dyn_strides(),
                  [](const auto& stride) { return sym::fixed_value(stride).has_value(); });
}

bool block_is_closed(
    const block_plan& block,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    std::unordered_set<instruction_ref> included;
    for(const auto* op : block.ops)
        included.insert(op->info.ins);
    auto owned_roots = get_owned_roots(block);

    auto boundary_reaches_block = [&](instruction_ref dependency) {
        std::unordered_set<instruction_ref> boundary_visited;
        return fix<bool>([&](auto self, instruction_ref current) -> bool {
            if(not boundary_visited.insert(current).second)
                return false;
            if(contains(included, current))
                return true;
            return any_of(current->inputs(), self);
        })(dependency);
    };

    std::unordered_map<instruction_ref, bool> dependency_cache;
    auto dependencies_are_closed = fix<bool>([&](auto self, instruction_ref current) -> bool {
        auto cached = dependency_cache.find(current);
        if(cached != dependency_cache.end())
            return cached->second;
        bool result = false;
        if(contains(included, current))
            result = true;
        else if(not freezer_roots_are_owned(current, owned_roots, info_for_instruction))
            result = not boundary_reaches_block(current);
        else
        {
            auto found = info_for_instruction.find(current);
            bool absorbable_freezer =
                found != info_for_instruction.end() and found->second->block.has_value() and
                found->second->info.freezer and current->name() == "reshape" and
                current->inputs().size() == 2;
            if(found != info_for_instruction.end() and found->second->block.has_value() and
               not absorbable_freezer)
                result = not boundary_reaches_block(current);
            else if(absorbable_dependency(current, included, info_for_instruction) or
                    owned_shape_transform_dependency(current, owned_roots, info_for_instruction))
                result = all_of(current->inputs(), self);
            else
                result = not boundary_reaches_block(current);
        }
        dependency_cache.emplace(current, result);
        return result;
    });

    for(const auto* op : block.ops)
    {
        const auto& info = op->info;
        const auto& args = info.ins->inputs();
        assert(args.size() == info.operands.size());
        for(std::size_t index = 0; index < args.size(); ++index)
        {
            if(contains(info.shape_input_indices, index))
                continue;
            auto source         = args.at(index);
            const auto& operand = info.operands.at(index);
            if(contains(included, source))
            {
                bool keeps_logical_slice =
                    source->get_shape().type() != shape::tuple_type and
                    (not operand.pad_value.has_value() or not operand.retained_slice_axes.empty());
                if(keeps_logical_slice)
                    return false;
                continue;
            }
            if(not dependencies_are_closed(source))
                return false;
        }
    }
    return true;
}

std::optional<std::vector<const root_spec*>>
merge_block_roots(const block_plan& target, const block_plan& source, std::size_t max_clones)
{
    std::vector<const root_spec*> result;
    result.reserve(target.roots.size() + source.roots.size());
    std::set_union(target.roots.begin(),
                   target.roots.end(),
                   source.roots.begin(),
                   source.roots.end(),
                   std::back_inserter(result),
                   [](const auto* x, const auto* y) { return x->name < y->name; });
    if(not roots_fit_clone_limit(result, max_clones))
        return std::nullopt;
    return result;
}

bool merge_block_into(
    block_plan& target,
    const block_plan& source,
    std::size_t max_clones,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    const std::unordered_set<sym::expr>& opaque_roots)
{
    if(any_of(opaque_roots, [&](const auto& root) {
           auto uses_root = [&](const block_plan& block) {
               return any_of(block.roots,
                             [&](const auto* candidate) { return candidate->root == root; });
           };
           return uses_root(target) != uses_root(source);
       }))
        return false;
    auto merged_roots = merge_block_roots(target, source, max_clones);
    if(not merged_roots.has_value())
        return false;
    block_plan result;
    result.ops = target.ops;
    result.ops.insert(result.ops.end(), source.ops.begin(), source.ops.end());
    result.roots = std::move(*merged_roots);
    if(not block_is_closed(result, info_for_instruction))
        return false;
    target = std::move(result);
    return true;
}

// Coalescing turns per-instruction specialization candidates into larger regions so connected
// operations share one select_module. A merge is accepted only when the combined region is closed
// over its dynamic dependencies and the cartesian product of root buckets stays within max_clones.
// Block index preserves topological order and empty ops marks a merged block.
std::optional<std::size_t> merge_blocks(
    std::vector<block_plan>& blocks,
    std::size_t x,
    std::size_t y,
    std::size_t max_clones,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    const std::unordered_set<sym::expr>& opaque_roots)
{
    const auto target = std::min(x, y);
    const auto source = std::max(x, y);
    if(target == source or blocks.at(source).ops.empty())
        return std::nullopt;
    if(not merge_block_into(
           blocks.at(target), blocks.at(source), max_clones, info_for_instruction, opaque_roots))
        return std::nullopt;
    for(auto* op : blocks.at(source).ops)
        op->block = target;
    blocks.at(source).ops.clear();
    return target;
}

void insert_block_connection(
    std::set<std::pair<std::size_t, std::size_t>>& connections,
    instruction_ref consumer,
    std::size_t input,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    auto consumer_info = info_for_instruction.find(consumer);
    if(consumer_info == info_for_instruction.end() or not consumer_info->second->block.has_value())
        return;
    auto producer_info = info_for_instruction.find(consumer->inputs().at(input));
    if(producer_info == info_for_instruction.end() or
       not producer_info->second->block.has_value() or
       producer_info->second->block == consumer_info->second->block)
        return;
    connections.insert({consumer_info->second->rank, input});
}

std::set<std::pair<std::size_t, std::size_t>> find_all_block_connections(
    const std::vector<symbolic_op_plan>& infos,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    std::set<std::pair<std::size_t, std::size_t>> result;
    for(const auto& plan : infos)
    {
        const auto& info = plan.info;
        for(auto input : range(info.ins->inputs().size()))
            insert_block_connection(result, info.ins, input, info_for_instruction);
    }
    return result;
}

std::set<std::pair<std::size_t, std::size_t>> find_block_connections(
    const block_plan& block,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    std::set<std::pair<std::size_t, std::size_t>> result;
    for(const auto* plan : block.ops)
    {
        const auto& info = plan->info;
        for(auto input : range(info.ins->inputs().size()))
            insert_block_connection(result, info.ins, input, info_for_instruction);
        for(auto output : info.ins->outputs())
        {
            const auto& inputs = output->inputs();
            for(auto input : range(inputs.size()))
                if(inputs.at(input) == info.ins)
                    insert_block_connection(result, output, input, info_for_instruction);
        }
    }
    return result;
}

void coalesce_connected_blocks(
    const std::vector<symbolic_op_plan>& infos,
    std::vector<block_plan>& blocks,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    std::set<std::pair<std::size_t, std::size_t>>& connections,
    std::size_t max_clones,
    const std::unordered_set<sym::expr>& opaque_roots)
{
    // Prefer producer-consumer merges. A successful merge can make an earlier rejected edge safe,
    // so revisit affected edges until no additional connected blocks can be combined.
    std::set<std::pair<std::size_t, std::size_t>> retries;
    while(not connections.empty())
    {
        auto connection = *connections.begin();
        connections.erase(connections.begin());
        auto [consumer_rank, input] = connection;
        const auto& consumer        = infos.at(consumer_rank).info.ins;
        const auto& producer        = consumer->inputs().at(input);
        const auto* consumer_info   = info_for_instruction.at(consumer);
        const auto* producer_info   = info_for_instruction.at(producer);
        auto merged                 = merge_blocks(blocks,
                                   *consumer_info->block,
                                   *producer_info->block,
                                   max_clones,
                                   info_for_instruction,
                                   opaque_roots);
        if(merged.has_value())
        {
            auto affected = find_block_connections(blocks.at(*merged), info_for_instruction);
            auto next     = affected.upper_bound(connection);
            retries.insert(affected.begin(), next);
            connections.insert(next, affected.end());
        }
        if(connections.empty())
            connections.swap(retries);
    }
}

void coalesce_independent_blocks(
    const std::vector<symbolic_op_plan>& infos,
    std::vector<block_plan>& blocks,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    std::size_t max_clones,
    const std::unordered_set<sym::expr>& opaque_roots,
    const std::unordered_set<instruction_ref>& pre_dyn_concat_instructions,
    const std::unordered_set<sym::expr>& data_roots)
{
    auto is_protected = [&](const block_plan& block) {
        return any_of(block.ops, [&](const auto* plan) {
            return contains(pre_dyn_concat_instructions, plan->info.ins);
        });
    };
    for(auto target : range(blocks.size()))
    {
        if(blocks.at(target).ops.empty())
            continue;
        for(auto source : range(target + 1, blocks.size()))
        {
            if(blocks.at(source).ops.empty())
                continue;
            bool target_protected = is_protected(blocks.at(target));
            bool source_protected = is_protected(blocks.at(source));
            if(target_protected != source_protected)
                continue;
            auto target_data_roots = get_block_data_roots(blocks.at(target), data_roots);
            auto source_data_roots = get_block_data_roots(blocks.at(source), data_roots);
            if(target_data_roots != source_data_roots and
               (not target_data_roots.empty() or not source_data_roots.empty()))
                continue;
            if(target_protected and not same_block_roots(blocks.at(target), blocks.at(source)))
                continue;
            auto merged = merge_blocks(
                blocks, target, source, max_clones, info_for_instruction, opaque_roots);
            if(merged.has_value())
            {
                auto connections = find_block_connections(blocks.at(*merged), info_for_instruction);
                coalesce_connected_blocks(
                    infos, blocks, info_for_instruction, connections, max_clones, opaque_roots);
                if(blocks.at(target).ops.empty())
                    break;
            }
        }
    }
}

std::vector<block_plan> discover_blocks(
    std::vector<symbolic_op_plan>& infos,
    const std::vector<root_spec>& roots,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    std::size_t max_clones,
    const std::unordered_set<sym::expr>& opaque_roots,
    const std::unordered_set<instruction_ref>& pre_dyn_concat_instructions,
    const std::unordered_set<sym::expr>& data_roots)
{
    // Start with one block per operation that needs padding or a static rewrite. Attach the root
    // symbols needed to specialize that operation, then coalesce both producer-consumer blocks and
    // independent blocks when they can safely share one set of clones.
    std::vector<block_plan> blocks;
    for(auto& plan : infos)
    {
        if(not can_specialize(plan))
            continue;
        auto required_roots = find_instruction_roots(plan.info, roots, max_clones);
        if(not required_roots.has_value())
            continue;
        plan.block = blocks.size();
        blocks.push_back({{&plan}, std::move(*required_roots)});
    }

    auto connections = find_all_block_connections(infos, info_for_instruction);
    coalesce_connected_blocks(
        infos, blocks, info_for_instruction, connections, max_clones, opaque_roots);
    coalesce_independent_blocks(infos,
                                blocks,
                                info_for_instruction,
                                max_clones,
                                opaque_roots,
                                pre_dyn_concat_instructions,
                                data_roots);

    blocks.erase(std::remove_if(blocks.begin(),
                                blocks.end(),
                                [](const auto& block) { return block.ops.empty(); }),
                 blocks.end());
    for(std::size_t block_index = 0; block_index < blocks.size(); ++block_index)
        for(auto* plan : blocks.at(block_index).ops)
            plan->block = block_index;
    return blocks;
}

struct pre_dyn_concat_analysis
{
    std::unordered_set<instruction_ref> instructions;
    std::unordered_set<sym::expr> data_roots;
};

pre_dyn_concat_analysis find_pre_dyn_concat_analysis(
    const root_registry& registry,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    pre_dyn_concat_analysis result;
    std::unordered_set<instruction_ref> visited;
    auto visit = fix<void>([&](auto self, instruction_ref ins) {
        if(not visited.insert(ins).second or starts_with(ins->name(), "@"))
            return;
        if(contains(registry.boundaries, ins))
        {
            const auto& s = ins->get_shape();
            if(s.symbolic())
                for(const auto& d : s.dyn_dims())
                    if(d.sym_expr.name() == "variable")
                    {
                        auto root = sym::as_symbol(d.sym_expr);
                        if(contains(registry.data_roots, root))
                            result.data_roots.insert(root);
                    }
            return;
        }
        auto info = info_for_instruction.find(ins);
        if(info != info_for_instruction.end())
            result.instructions.insert(ins);
        for(auto input : ins->inputs())
            self(input);
    });
    for(const auto& [slice, concat] : registry.dyn_concat_boundaries)
    {
        (void)slice;
        for(auto input : concat->inputs())
            visit(input);
    }
    return result;
}

void order_blocks_for_staged_roots(
    std::vector<block_plan>& blocks,
    const root_registry& registry,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    if(blocks.size() < 2)
        return;

    std::vector<std::unordered_set<std::size_t>> dependencies(blocks.size());
    for(std::size_t consumer = 0; consumer < blocks.size(); ++consumer)
    {
        std::unordered_set<instruction_ref> block_visited;
        auto collect_block_producers = fix<void>([&](auto self, instruction_ref ins) {
            if(not block_visited.insert(ins).second)
                return;
            auto info = info_for_instruction.find(ins);
            if(info != info_for_instruction.end())
            {
                if(info->second->block.has_value())
                {
                    auto producer = *info->second->block;
                    if(producer != consumer)
                        dependencies.at(consumer).insert(producer);
                    return;
                }
                if(info->second->root_source)
                    return;
            }
            for(auto input : ins->inputs())
                self(input);
        });
        for(const auto* plan : blocks.at(consumer).ops)
            for(auto input : plan->info.ins->inputs())
                collect_block_producers(input);

        for(const auto* root : blocks.at(consumer).roots)
        {
            auto source = registry.sources.find(root->root);
            if(source == registry.sources.end())
                continue;
            for(auto input : source->second->inputs())
                collect_block_producers(input);
        }
    }

    std::vector<bool> emitted(blocks.size(), false);
    std::vector<block_plan> ordered;
    ordered.reserve(blocks.size());
    for(std::size_t position = 0; position < blocks.size(); ++position)
    {
        auto indices = range(blocks.size());
        auto ready   = std::find_if(indices.begin(), indices.end(), [&](auto block) {
            return not emitted.at(block) and all_of(dependencies.at(block), [&](auto dependency) {
                return emitted.at(dependency);
            });
        });
        if(ready == indices.end())
        {
            auto blocked = std::find(emitted.begin(), emitted.end(), false);
            assert(blocked != emitted.end());
            auto block = static_cast<std::size_t>(std::distance(emitted.begin(), blocked));
            auto dependency =
                std::find_if(dependencies.at(block).begin(),
                             dependencies.at(block).end(),
                             [&](auto candidate) { return not emitted.at(candidate); });
            assert(dependency != dependencies.at(block).end());
            MIGRAPHX_THROW("SPLIT_SYM_DIM: cyclic staged-root dependency between blocks " +
                           std::to_string(block) + " and " + std::to_string(*dependency));
        }
        emitted.at(*ready) = true;
        ordered.push_back(std::move(blocks.at(*ready)));
    }
    blocks = std::move(ordered);
    for(std::size_t block = 0; block < blocks.size(); ++block)
        for(auto* plan : blocks.at(block).ops)
            plan->block = block;
}

instruction_ref
add_or_reuse_pad(module& m,
                 const operation& pad_op,
                 instruction_ref input,
                 std::unordered_map<instruction_ref, std::vector<instruction_ref>>& cache)
{
    auto& candidates = cache[input];
    auto it = std::find_if(candidates.begin(), candidates.end(), [&](instruction_ref candidate) {
        return candidate->get_operator() == pad_op;
    });
    if(it != candidates.end())
        return *it;
    auto result = m.add_instruction(pad_op, input);
    candidates.push_back(result);
    return result;
}

bool needs_fixed_retarget(const shape& s,
                          const std::unordered_map<sym::expr, sym::expr>& substitutions)
{
    if(not s.symbolic())
        return false;
    return any_of(s.dyn_dims(),
                  [&](const auto& d) {
                      return d.is_fixed() and d.sym_expr.subs(substitutions) != d.sym_expr;
                  }) or
           any_of(s.dyn_strides(),
                  [&](const auto& stride) { return stride.subs(substitutions) != stride; });
}

void prepare_clone_infos(
    std::vector<symbolic_op_plan>& infos,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    const std::vector<root_spec>& roots)
{
    std::unordered_map<sym::expr, sym::expr> target_substitutions;
    for(const auto& root : roots)
        target_substitutions.emplace(root.root, root.target_symbol);

    for(auto& plan : infos)
    {
        if(not plan.block.has_value())
            continue;
        const auto& info  = plan.info;
        const auto& args  = info.ins->inputs();
        auto input_shapes = info.get_input_shapes();
        assert(input_shapes.size() == args.size());
        assert(info.operands.size() == args.size());
        plan.clone_inputs.clear();
        plan.clone_inputs.reserve(args.size());
        for(auto input_index : range(args.size()))
        {
            if(contains(info.shape_input_indices, input_index))
                continue;
            auto operand = info.operands.at(input_index);
            auto source  = args.at(input_index);
            sliced_value input{source, {}};
            auto source_info             = info_for_instruction.find(source);
            const bool source_is_planned = source_info != info_for_instruction.end() and
                                           source_info->second->block.has_value();
            const bool source_in_same_block =
                source_is_planned and source_info->second->block == plan.block;
            const bool source_in_other_block = source_is_planned and not source_in_same_block;

            // Start from the cross-block form; internal edges are simplified below.
            if(source_is_planned)
                input.slice_axes = source_info->second->info.output_symbolic_axes;

            // Fixed symbolic dimensions and strides still require static clone metadata.
            bool emit_pad =
                operand.pad_value.has_value() or
                needs_fixed_retarget(input_shapes.at(input_index), target_substitutions);
            if(source_in_same_block and source->get_shape().type() == shape::tuple_type)
            {
                input.slice_axes.clear();
                emit_pad = false;
            }
            else if(source_in_same_block and operand.pad_value.has_value())
            {
                assert(operand.retained_slice_axes.empty());
                // Internal edges already carry bucket-sized values.
                input.slice_axes.clear();
                emit_pad = false;
            }
            else if(source_in_other_block and operand.pad_value.has_value() and
                    not operand.retained_slice_axes.empty())
            {
                // Cross-block edges retain only axes where padding can affect the consumer.
                std::vector<std::size_t> kept_axes;
                std::copy_if(
                    input.slice_axes.begin(),
                    input.slice_axes.end(),
                    std::back_inserter(kept_axes),
                    [&](auto axis) { return contains(operand.retained_slice_axes, axis); });
                if(kept_axes.empty())
                {
                    input.slice_axes.clear();
                    emit_pad = false;
                }
                else
                    input.slice_axes = std::move(kept_axes);
            }
            // A retarget-only pad has no reachable padded region, so its fill value is irrelevant.
            if(emit_pad)
                operand.pad_value = operand.pad_value.value_or(0.0f);
            else
                operand.pad_value.reset();
            plan.clone_inputs.push_back({std::move(input), std::move(operand)});
        }
        plan.dispatch_output = substitute_shape(info.get_output_shape(), target_substitutions);
    }
}

shape clone_parameter_shape(
    const shape& s,
    const std::unordered_map<sym::expr, shape::dynamic_dimension::interval>& subranges,
    const std::unordered_map<sym::expr, std::size_t>& freeze)
{
    if(not s.symbolic())
        return s;
    std::unordered_map<sym::expr, sym::expr> substitutions;
    for(const auto& [root, interval] : subranges)
    {
        substitutions.emplace(root, sym::var(root.to_string(), {interval.min, interval.max}));
    }
    auto result = substitute_shape(s, substitutions);
    std::unordered_map<sym::expr, sym::expr> frozen_symbols;
    for(const auto& [symbol, value] : freeze)
        if(not contains(subranges, symbol))
            frozen_symbols.emplace(symbol, sym::lit(value));
    auto dimensions = result.dyn_dims();
    std::transform(dimensions.begin(), dimensions.end(), dimensions.begin(), [&](const auto& d) {
        return shape::dynamic_dimension{d.sym_expr.subs(frozen_symbols)};
    });
    result = {result.type(), std::move(dimensions), result.dyn_strides()};
    if(s.is_fixed() and all_of(result.dyn_strides(), [](const auto& stride) {
           return sym::fixed_value(stride).has_value();
       }))
        return result.to_static();
    return result;
}

struct clone_output_case
{
    std::unordered_map<sym::expr, std::size_t> freeze;
    std::vector<shape> outputs;
    std::vector<std::size_t> output_indices;
};

bool represents_clone_outputs(const shape& candidate,
                              const std::vector<clone_output_case>& clone_outputs,
                              std::size_t output_index)
{
    return all_of(clone_outputs, [&](const auto& clone_output) {
        const auto& output = clone_output.outputs.at(output_index);
        auto expected      = candidate.to_static(clone_output.freeze);
        if(expected.type() != output.type() or expected.lens() != output.lens())
            return false;
        if(expected.elements() == 0)
            return true;
        for(std::size_t axis = 0; axis < expected.ndim(); ++axis)
        {
            if(expected.lens()[axis] > 1 and expected.strides()[axis] != output.strides()[axis])
                return false;
        }
        return true;
    });
}

shape dispatch_shape_for_clones(const shape& planned,
                                const std::vector<clone_output_case>& clone_outputs,
                                std::size_t output_index)
{
    assert(planned.symbolic());
    assert(not clone_outputs.empty());
    if(represents_clone_outputs(planned, clone_outputs, output_index))
        return planned;

    for(const auto& clone_output : clone_outputs)
    {
        auto actual_layout =
            clone_output.outputs.at(output_index).with_lens(planned.type(), planned.dyn_dims());
        if(represents_clone_outputs(actual_layout, clone_outputs, output_index))
            return actual_layout;
    }

    std::vector<shape> outputs;
    std::transform(clone_outputs.begin(),
                   clone_outputs.end(),
                   std::back_inserter(outputs),
                   [&](const auto& clone_output) { return clone_output.outputs.at(output_index); });
    MIGRAPHX_THROW("SPLIT_SYM_DIM: planned dispatch shape " + to_string(planned) +
                   " does not represent clone outputs " + to_string_range(outputs));
}

bool is_shape_input(const symbolic_op_plan* plan, std::size_t index)
{
    return plan != nullptr and plan->info.freezer and
           contains(plan->info.shape_input_indices, index);
}

std::vector<sliced_value> clone_inputs_for(
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    instruction_ref ins)
{
    auto found = info_for_instruction.find(ins);
    if(found != info_for_instruction.end() and found->second->block.has_value())
    {
        std::vector<sliced_value> result;
        result.reserve(found->second->clone_inputs.size());
        std::transform(found->second->clone_inputs.begin(),
                       found->second->clone_inputs.end(),
                       std::back_inserter(result),
                       [](const auto& input) { return input.clone_value; });
        return result;
    }
    std::vector<sliced_value> result;
    auto inputs        = ins->inputs();
    const auto* plan   = found == info_for_instruction.end() ? nullptr : found->second;
    auto input_indices = range(inputs.size());
    for(auto&& [index, input] : views::zip(input_indices, inputs))
        if(not is_shape_input(plan, index))
            result.push_back({input, {}});
    return result;
}

struct runtime_cache
{
    std::unordered_map<sym::expr, instruction_ref> extents;
    std::unordered_map<std::size_t, instruction_ref> indices;
    std::map<std::pair<shape::type_t, fill_kind>, instruction_ref> fills;
};

instruction_ref resolved_extent(module& m,
                                const sym::expr& expression,
                                const std::vector<instruction_ref>& sources,
                                runtime_cache& cache)
{
    auto cached = cache.extents.find(expression);
    if(cached != cache.extents.end())
        return cached->second;
    auto result =
        m.add_instruction(make_op("eval_expr_from_shape",
                                  {{"expressions", to_value(std::vector<sym::expr>{expression})}}),
                          sources);
    return cache.extents.emplace(expression, result).first->second;
}

instruction_ref index_literal(module& m, std::size_t n, runtime_cache& cache)
{
    auto cached = cache.indices.find(n);
    if(cached != cache.indices.end())
        return cached->second;
    std::vector<int64_t> indices(n);
    std::iota(indices.begin(), indices.end(), int64_t{0});
    auto result = m.add_literal(literal{shape{shape::int64_type, {n}}, indices});
    return cache.indices.emplace(n, result).first->second;
}

instruction_ref fill_literal(module& m, shape::type_t type, fill_kind fill, runtime_cache& cache)
{
    auto key    = std::make_pair(type, fill);
    auto cached = cache.fills.find(key);
    if(cached != cache.fills.end())
        return cached->second;
    auto result = m.add_literal(literal{shape{type, {1}}, std::vector<float>{fill_value(fill)}});
    return cache.fills.emplace(key, result).first->second;
}

instruction_ref
add_runtime_mask(module& m,
                 instruction_ref input,
                 const axis_mask& mask,
                 const std::vector<instruction_ref>& sources,
                 const std::unordered_map<sym::expr, sym::expr>& fixed_substitutions,
                 runtime_cache& cache)
{
    const auto& s = input->get_shape();
    assert(not s.dynamic());
    assert(mask.axis < s.ndim());
    auto lens = s.lens();

    auto index  = m.add_instruction(make_op("broadcast", {{"axis", mask.axis}, {"out_lens", lens}}),
                                   index_literal(m, lens[mask.axis], cache));
    auto extent = m.add_instruction(
        make_op("multibroadcast", {{"out_lens", lens}}),
        resolved_extent(m, mask.extent.subs(fixed_substitutions), sources, cache));
    auto valid = m.add_instruction(make_op("convert", {{"target_type", shape::bool_type}}),
                                   m.add_instruction(make_op("less"), index, extent));
    auto fill  = m.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}),
                                  fill_literal(m, s.type(), mask.fill, cache));
    return m.add_instruction(make_op("where"), valid, input, fill);
}

struct block_input
{
    sliced_value clone_value;
    instruction_ref select_input;
};

struct block_frame
{
    std::vector<instruction_ref> body;
    std::vector<sliced_value> outputs;
    std::vector<block_input> inputs;
    std::map<std::string, std::size_t> params;
    std::vector<std::size_t> literals;
    std::vector<std::size_t> extent_sources;
};

struct deduplicated_block_outputs
{
    std::vector<instruction_ref> outputs;
    std::vector<std::size_t> indices;
};

deduplicated_block_outputs deduplicate_block_outputs(const std::vector<instruction_ref>& outputs)
{
    deduplicated_block_outputs result;
    std::transform(outputs.begin(),
                   outputs.end(),
                   std::back_inserter(result.indices),
                   [&](instruction_ref output) {
                       auto found = std::find(result.outputs.begin(), result.outputs.end(), output);
                       if(found != result.outputs.end())
                           return static_cast<std::size_t>(
                               std::distance(result.outputs.begin(), found));
                       result.outputs.push_back(output);
                       return result.outputs.size() - 1;
                   });
    return result;
}

sliced_value full_output_for(
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    instruction_ref source)
{
    auto found = info_for_instruction.find(source);
    if(found == info_for_instruction.end())
        MIGRAPHX_THROW("SPLIT_SYM_DIM: missing symbolic information for block output " +
                       source->name());
    sliced_value result{source, {}};
    result.slice_axes = found->second->info.output_symbolic_axes;
    return result;
}

void add_required_output(
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    block_frame& frame,
    sliced_value output)
{
    if(output.slice_axes.empty())
        output = full_output_for(info_for_instruction, output.source);
    if(not contains(frame.outputs, output))
        frame.outputs.push_back(std::move(output));
}

std::size_t add_block_input(std::vector<block_input>& inputs, sliced_value clone_value)
{
    auto found = std::find_if(inputs.begin(), inputs.end(), [&](const auto& input) {
        return input.clone_value == clone_value;
    });
    if(found != inputs.end())
        return static_cast<std::size_t>(std::distance(inputs.begin(), found));
    auto select_input = clone_value.source;
    inputs.push_back({std::move(clone_value), select_input});
    return inputs.size() - 1;
}

struct dependency_absorption
{
    const std::unordered_set<instruction_ref>& planned;
    const std::unordered_set<sym::expr>& owned_roots;
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction;
    std::unordered_map<instruction_ref, bool> cache;

    bool rejects_freezer(instruction_ref ins) const
    {
        return absorbable_dependency(ins, planned, info_for_instruction) and
               not freezer_roots_are_owned(ins, owned_roots, info_for_instruction);
    }

    bool is_absorbable(instruction_ref ins) const
    {
        return absorbable_dependency(ins, planned, info_for_instruction) or
               owned_shape_transform_dependency(ins, owned_roots, info_for_instruction);
    }

    bool operator()(instruction_ref ins)
    {
        auto cached = cache.find(ins);
        if(cached != cache.end())
            return cached->second;
        if(contains(planned, ins))
            return cache.emplace(ins, true).first->second;
        auto found = info_for_instruction.find(ins);
        if(found != info_for_instruction.end() and found->second->block.has_value())
        {
            const auto& info        = found->second->info;
            bool absorbable_freezer = info.freezer and ins->name() == "reshape" and
                                      ins->inputs().size() == 2 and not rejects_freezer(ins);
            if(not absorbable_freezer)
                return cache.emplace(ins, false).first->second;
        }
        if(not is_absorbable(ins) or rejects_freezer(ins))
            return cache.emplace(ins, false).first->second;
        auto inputs = clone_inputs_for(info_for_instruction, ins);
        bool result = all_of(inputs, [&](const auto& input) {
            return not is_absorbable(input.source) or (*this)(input.source);
        });
        cache.emplace(ins, result);
        return result;
    }
};

std::unordered_set<instruction_ref> find_required_block_outputs(
    const std::vector<block_plan>& blocks,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    std::unordered_set<instruction_ref> result;
    for(std::size_t block_number = 0; block_number < blocks.size(); ++block_number)
    {
        const auto& block = blocks.at(block_number);
        std::unordered_set<instruction_ref> planned;
        std::transform(block.ops.begin(),
                       block.ops.end(),
                       std::inserter(planned, planned.end()),
                       [](const auto* plan) { return plan->info.ins; });
        auto owned_roots = get_owned_roots(block);
        dependency_absorption can_absorb{planned, owned_roots, info_for_instruction};

        std::vector<instruction_ref> boundaries;
        std::unordered_set<instruction_ref> visited;
        auto visit = fix<void>([&](auto self, instruction_ref ins) {
            if(not visited.insert(ins).second)
                return;
            for(const auto& input : clone_inputs_for(info_for_instruction, ins))
            {
                if(can_absorb(input.source))
                    self(input.source);
                else
                    boundaries.push_back(input.source);
            }
        });
        for(const auto* plan : block.ops)
            visit(plan->info.ins);

        std::unordered_set<instruction_ref> dependency_visited;
        auto collect_block_outputs = fix<void>([&](auto self, instruction_ref ins) {
            if(not dependency_visited.insert(ins).second)
                return;
            auto found = info_for_instruction.find(ins);
            if(found != info_for_instruction.end())
            {
                if(found->second->block.has_value())
                {
                    if(found->second->block != block_number)
                        result.insert(ins);
                    return;
                }
                if(found->second->root_source)
                    return;
            }
            for(auto input : ins->inputs())
                self(input);
        });
        for(auto boundary : boundaries)
            collect_block_outputs(boundary);
    }
    return result;
}

void collect_frame_outputs(
    const module& m,
    const block_plan& block,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    const std::unordered_set<instruction_ref>& planned_instructions,
    const std::unordered_set<instruction_ref>& body_instructions,
    const std::unordered_set<instruction_ref>& required_block_outputs,
    block_frame& frame)
{
    for(auto output : m.get_returns())
        if(contains(planned_instructions, output))
            add_required_output(info_for_instruction, frame, {output, {}});
    for(auto output : required_block_outputs)
        if(contains(body_instructions, output))
            add_required_output(info_for_instruction, frame, {output, {}});

    std::vector<instruction_ref> external_consumers;
    for(auto body_ins : body_instructions)
        for(auto consumer : body_ins->outputs())
            if(not contains(body_instructions, consumer) and
               not contains(external_consumers, consumer))
                external_consumers.push_back(consumer);
    for(auto consumer : external_consumers)
        for(auto input : clone_inputs_for(info_for_instruction, consumer))
            if(contains(planned_instructions, input.source))
                add_required_output(info_for_instruction, frame, std::move(input));
}

void add_frame_root_inputs(block_frame& frame,
                           const block_plan& block,
                           const std::unordered_map<sym::expr, instruction_ref>& root_sources)
{
    for(const auto* root : block.roots)
    {
        if(not contains(root_sources, root->root))
            MIGRAPHX_THROW("SPLIT_SYM_DIM: no parameter resolves block root " + root->name);
        auto input_index = add_block_input(frame.inputs, {root_sources.at(root->root), {}});
        if(not contains(frame.extent_sources, input_index))
            frame.extent_sources.push_back(input_index);
    }
}

void name_frame_inputs(block_frame& frame, const module& m, std::size_t block_number)
{
    std::unordered_set<std::string> used_names;
    for(const auto& name : m.get_parameter_names())
        used_names.insert(name);
    const std::string input_prefix = "#split_sym_dim_input_";
    std::size_t generated_suffix   = 0;
    for(auto input_index : range(frame.inputs.size()))
    {
        auto source = frame.inputs.at(input_index).clone_value.source;
        if(source->name() == "@param")
        {
            auto parameter_name =
                source->get_operator().to_value().at("parameter").to<std::string>();
            frame.params.emplace(std::move(parameter_name), input_index);
            continue;
        }
        if(source->name() == "@literal")
        {
            frame.literals.push_back(input_index);
            ++generated_suffix;
            continue;
        }
        auto name =
            input_prefix + std::to_string(block_number) + "_" + std::to_string(generated_suffix++);
        while(not used_names.insert(name).second)
            name = input_prefix + std::to_string(block_number) + "_" +
                   std::to_string(generated_suffix++);
        frame.params.emplace(std::move(name), input_index);
    }
    if(frame.params.size() + frame.literals.size() != frame.inputs.size())
        MIGRAPHX_THROW("SPLIT_SYM_DIM: failed to collect every block input");
}

void collect_block_body(
    instruction_ref ins,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    dependency_absorption& can_absorb,
    std::size_t block_number,
    std::unordered_set<instruction_ref>& body_instructions,
    block_frame& frame)
{
    if(not body_instructions.insert(ins).second)
        return;
    auto inputs = clone_inputs_for(info_for_instruction, ins);
    for(const auto& input : inputs)
        if(can_absorb(input.source))
            collect_block_body(input.source,
                               info_for_instruction,
                               can_absorb,
                               block_number,
                               body_instructions,
                               frame);
    frame.body.push_back(ins);
    for(auto input : inputs)
    {
        auto source_info            = info_for_instruction.find(input.source);
        const bool rejected_freezer = can_absorb.rejects_freezer(input.source);
        if(rejected_freezer and input.slice_axes.empty() and
           source_info != info_for_instruction.end() and source_info->second->block.has_value() and
           source_info->second->block != block_number)
        {
            input = full_output_for(info_for_instruction, input.source);
        }
        add_block_input(frame.inputs, std::move(input));
    }
}

std::optional<block_frame> find_block_frame(
    const module& m,
    const block_plan& block,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    std::size_t block_number,
    const std::unordered_map<sym::expr, instruction_ref>& root_sources,
    const std::unordered_set<instruction_ref>& required_block_outputs)
{
    std::unordered_set<instruction_ref> planned_instructions;
    std::transform(block.ops.begin(),
                   block.ops.end(),
                   std::inserter(planned_instructions, planned_instructions.end()),
                   [](const auto* plan) { return plan->info.ins; });

    block_frame result;
    std::unordered_set<instruction_ref> body_instructions;
    auto owned_roots = get_owned_roots(block);
    dependency_absorption can_absorb{planned_instructions, owned_roots, info_for_instruction};
    for(const auto* plan : block.ops)
        collect_block_body(plan->info.ins,
                           info_for_instruction,
                           can_absorb,
                           block_number,
                           body_instructions,
                           result);

    if(body_instructions.empty())
        return std::nullopt;

    result.inputs.erase(std::remove_if(result.inputs.begin(),
                                       result.inputs.end(),
                                       [&](const auto& input) {
                                           return contains(body_instructions,
                                                           input.clone_value.source) and
                                                  input.clone_value.slice_axes.empty();
                                       }),
                        result.inputs.end());

    collect_frame_outputs(m,
                          block,
                          info_for_instruction,
                          planned_instructions,
                          body_instructions,
                          required_block_outputs,
                          result);
    if(result.outputs.empty())
        return std::nullopt;

    add_frame_root_inputs(result, block, root_sources);
    name_frame_inputs(result, m, block_number);
    return result;
}

instruction_ref
find_cloned_input(const sliced_value& input,
                  const std::unordered_map<instruction_ref, instruction_ref>& clone_map,
                  const std::vector<std::pair<sliced_value, instruction_ref>>& input_values)
{
    if(input.slice_axes.empty())
        return clone_map.at(input.source);
    auto found = std::find_if(input_values.begin(), input_values.end(), [&](const auto& value) {
        return value.first == input;
    });
    assert(found != input_values.end());
    return found->second;
}

bool only_used_as_slice_metadata(instruction_ref ins)
{
    const auto& outputs = ins->outputs();
    return not outputs.empty() and std::all_of(outputs.begin(), outputs.end(), [&](auto output) {
        return contains({"slice", "dyn_slice"}, output->name()) and output->inputs().front() != ins;
    });
}

struct clone_context
{
    module& clone_module;
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction;
    std::unordered_map<instruction_ref, instruction_ref>& clone_map;
    const std::vector<std::pair<sliced_value, instruction_ref>>& input_values;
    const std::unordered_map<sym::expr, std::size_t>& freeze;
    const std::vector<instruction_ref>& runtime_extent_sources;
    const std::unordered_map<sym::expr, sym::expr>& fixed_substitutions;
    runtime_cache cache;
    std::unordered_map<instruction_ref, std::vector<instruction_ref>> reusable_pads;

    clone_context(module& clone,
                  const std::unordered_map<instruction_ref, const symbolic_op_plan*>& infos,
                  std::unordered_map<instruction_ref, instruction_ref>& clones,
                  const std::vector<std::pair<sliced_value, instruction_ref>>& inputs,
                  const std::unordered_map<sym::expr, std::size_t>& frozen_values,
                  const std::vector<instruction_ref>& runtime_sources,
                  const std::unordered_map<sym::expr, sym::expr>& substitutions)
        : clone_module(clone),
          info_for_instruction(infos),
          clone_map(clones),
          input_values(inputs),
          freeze(frozen_values),
          runtime_extent_sources(runtime_sources),
          fixed_substitutions(substitutions)
    {
    }

    instruction_ref emit(instruction_ref source)
    {
        auto found                         = info_for_instruction.find(source);
        const symbolic_op_plan* clone_info = nullptr;
        if(found != info_for_instruction.end() and found->second->block.has_value())
            clone_info = found->second;
        auto source_inputs = clone_inputs_for(info_for_instruction, source);
        std::vector<instruction_ref> args;
        std::transform(source_inputs.begin(),
                       source_inputs.end(),
                       std::back_inserter(args),
                       [&](const sliced_value& input) {
                           return find_cloned_input(input, clone_map, input_values);
                       });
        if(clone_info != nullptr)
        {
            const auto& clone_inputs = clone_info->clone_inputs;
            assert(clone_inputs.size() == args.size());
            for(std::size_t index = 0; index < clone_inputs.size(); ++index)
                if(clone_inputs.at(index).operand.pad_value.has_value())
                    args.at(index) = add_or_reuse_pad(
                        clone_module,
                        make_op("fixed_pad",
                                {{"value", *clone_inputs.at(index).operand.pad_value}}),
                        args.at(index),
                        reusable_pads);
            for(std::size_t index = 0; index < clone_inputs.size(); ++index)
                for(const auto& mask : clone_inputs.at(index).operand.masks)
                    args.at(index) = add_runtime_mask(clone_module,
                                                      args.at(index),
                                                      mask,
                                                      runtime_extent_sources,
                                                      fixed_substitutions,
                                                      cache);
        }

        op_freezer freezer;
        if(clone_info != nullptr)
            freezer = clone_info->info.freezer;
        else if(source->get_shape().dynamic())
        {
            // Every absorbed dynamic instruction is symbolic, so it has an analysis entry.
            assert(found != info_for_instruction.end());
            freezer = found->second->info.freezer;
        }

        instruction_ref clone;
        if(freezer)
            clone = freezer(clone_module, source, args, freeze);
        else
        {
            clone =
                clone_module.add_instruction(source->get_operator(), args, source->module_inputs());
        }

        if(clone->get_shape().dynamic())
            MIGRAPHX_THROW("SPLIT_SYM_DIM: cloned " + source->name() +
                           " is not fully static: " + to_string(clone->get_shape()));
        clone_map[source] = clone;
        if(not source->get_debug_symbols().empty())
            clone_module.add_debug_symbols(clone, source->get_debug_symbols());
        return clone;
    }
};

struct fold_fixed_clone_evaluations : match::supports_dynamic_shapes
{
    const std::unordered_map<sym::expr, std::size_t>& fixed_runtime_values;

    auto matcher() const { return match::name("eval_expr_from_shape")(); }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins = mr.result;
        if(only_used_as_slice_metadata(ins))
            return;

        auto expressions =
            from_value<std::vector<sym::expr>>(ins->get_operator().to_value().at("expressions"));
        std::unordered_set<sym::expr> required;
        for(const auto& expression : expressions)
        {
            auto variables = sym::find_variables(expression);
            required.merge(variables);
        }
        if(required.empty() or any_of(required, [&](const auto& variable) {
               return not contains(fixed_runtime_values, variable);
           }))
            return;

        std::vector<int64_t> values;
        values.reserve(expressions.size());
        std::transform(expressions.begin(),
                       expressions.end(),
                       std::back_inserter(values),
                       [&](const auto& expression) {
                           return static_cast<int64_t>(expression.eval_uint(fixed_runtime_values));
                       });
        m.replace_instruction(
            ins, m.add_literal(literal{shape{shape::int64_type, {values.size()}}, values}));
    }
};

struct clone_build
{
    module clone;
    clone_output_case output_case;
};

// Materialize one block specialization. Parameters retain the runtime subrange accepted by this
// clone, while every operation in the body is emitted for the bucket's fixed target extents.
// Inputs are padded or masked as planned, and operators such as dyn_slice and symbolic broadcast
// are rewritten to static forms. No dynamic operation may remain in the emitted clone body.
clone_build build_clone(
    const std::string& name,
    const block_frame& frame,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    const std::unordered_map<sym::expr, std::size_t>& freeze,
    const std::unordered_map<sym::expr, shape::dynamic_dimension::interval>& subranges,
    const std::unordered_map<sym::expr, sym::expr>& fixed_substitutions)
{
    module clone_module{name};
    std::unordered_map<instruction_ref, instruction_ref> clone_map;
    std::vector<std::pair<sliced_value, instruction_ref>> input_values;
    for(const auto& [parameter_name, input_index] : frame.params)
    {
        const auto& input = frame.inputs.at(input_index);
        auto parameter    = clone_module.add_parameter(
            parameter_name,
            clone_parameter_shape(input.select_input->get_shape(), subranges, freeze));
        if(input.clone_value.slice_axes.empty())
            clone_map[input.clone_value.source] = parameter;
        else
            input_values.emplace_back(input.clone_value, parameter);
    }
    for(auto input_index : frame.literals)
    {
        auto source       = frame.inputs.at(input_index).clone_value.source;
        clone_map[source] = clone_module.add_literal(source->get_literal());
    }

    std::unordered_map<sym::expr, std::size_t> fixed_runtime_values;
    for(const auto& [root, interval] : subranges)
        if(interval.min == interval.max)
            fixed_runtime_values.emplace(root, interval.min);

    std::vector<instruction_ref> runtime_extent_sources;
    std::transform(frame.extent_sources.begin(),
                   frame.extent_sources.end(),
                   std::back_inserter(runtime_extent_sources),
                   [&](std::size_t input_index) {
                       return find_cloned_input(
                           frame.inputs.at(input_index).clone_value, clone_map, input_values);
                   });
    clone_context context{clone_module,
                          info_for_instruction,
                          clone_map,
                          input_values,
                          freeze,
                          runtime_extent_sources,
                          fixed_substitutions};
    for(auto source : frame.body)
        context.emit(source);
    match::find_matches(clone_module,
                        fold_fixed_clone_evaluations{.fixed_runtime_values = fixed_runtime_values});

    std::vector<instruction_ref> logical_outputs;
    std::transform(frame.outputs.begin(),
                   frame.outputs.end(),
                   std::back_inserter(logical_outputs),
                   [&](const sliced_value& output) { return clone_map.at(output.source); });
    clone_module.add_return(logical_outputs);
    run_passes(clone_module, {eliminate_convert{}, eliminate_identity{}, dead_code_elimination{}});

    auto deduplicated_outputs = deduplicate_block_outputs(clone_module.get_returns());
    if(clone_module.get_returns().size() > 1)
    {
        auto canonical_clone = clone_module;
        run_passes(canonical_clone, {eliminate_common_subexpression{}, dead_code_elimination{}});
        deduplicated_outputs = deduplicate_block_outputs(canonical_clone.get_returns());
    }
    auto physical_indices = range(deduplicated_outputs.outputs.size());
    std::vector<instruction_ref> clone_outputs;
    std::transform(physical_indices.begin(),
                   physical_indices.end(),
                   std::back_inserter(clone_outputs),
                   [&](std::size_t physical_index) {
                       auto logical = std::find(deduplicated_outputs.indices.begin(),
                                                deduplicated_outputs.indices.end(),
                                                physical_index);
                       assert(logical != deduplicated_outputs.indices.end());
                       auto logical_index = static_cast<std::size_t>(
                           std::distance(deduplicated_outputs.indices.begin(), logical));
                       return clone_module.get_returns().at(logical_index);
                   });
    if(any_of(clone_outputs, [](instruction_ref output) { return output->get_shape().dynamic(); }))
        MIGRAPHX_THROW("SPLIT_SYM_DIM: clone output is not fully static");
    clone_module.replace_return(clone_outputs);
    run_passes(clone_module, {dead_code_elimination{}});
    std::vector<shape> output_shapes;
    std::transform(clone_outputs.begin(),
                   clone_outputs.end(),
                   std::back_inserter(output_shapes),
                   [](instruction_ref output) { return output->get_shape(); });
    if(none_of(clone_module, [](const auto& ins) {
           return ins.name() == "eval_expr_from_shape" and not ins.outputs().empty();
       }))
    {
        auto static_clone = clone_module;
        for(auto parameter : static_clone.get_parameters())
        {
            const auto& s = parameter->get_shape();
            if(s.symbolic() and s.is_fixed() and all_of(s.dyn_strides(), [](const auto& stride) {
                   return sym::fixed_value(stride).has_value();
               }))
                instruction::replace(
                    parameter, parameter->get_operator(), s.to_static(), parameter->inputs());
        }
        if(static_clone.get_output_shapes() == output_shapes)
            clone_module = std::move(static_clone);
    }
    return {std::move(clone_module),
            {freeze, std::move(output_shapes), std::move(deduplicated_outputs.indices)}};
}

struct replacement_resolution
{
    std::unordered_map<instruction_ref, instruction_ref> memo;
};

instruction_ref resolve_replacement(
    module& m,
    instruction_ref source,
    std::unordered_map<instruction_ref, instruction_ref>& replacements,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    replacement_resolution& resolution,
    std::optional<std::size_t> consumer_block = std::nullopt,
    bool allow_unresolved_block               = false)
{
    auto cached = resolution.memo.find(source);
    if(cached != resolution.memo.end())
        return cached->second;
    auto found = replacements.find(source);
    if(found != replacements.end())
    {
        resolution.memo.emplace(source, found->second);
        return found->second;
    }
    auto info = info_for_instruction.find(source);
    if(info != info_for_instruction.end() and info->second->block.has_value())
    {
        if(allow_unresolved_block)
        {
            resolution.memo.emplace(source, source);
            return source;
        }
        if(consumer_block.has_value())
            MIGRAPHX_THROW("SPLIT_SYM_DIM: frame consumer block " +
                           std::to_string(*consumer_block) +
                           " reached unspecialized producer block " +
                           std::to_string(*info->second->block) + " through " + source->name());
        MIGRAPHX_THROW("SPLIT_SYM_DIM: block dependency was not specialized before use");
    }

    auto args    = source->inputs();
    bool changed = false;
    for(auto& arg : args)
    {
        auto replacement = resolve_replacement(m,
                                               arg,
                                               replacements,
                                               info_for_instruction,
                                               resolution,
                                               consumer_block,
                                               allow_unresolved_block);
        if(replacement == arg)
            continue;
        arg     = replacement;
        changed = true;
    }
    if(not changed)
    {
        resolution.memo.emplace(source, source);
        return source;
    }

    auto result = m.add_instruction(source->get_operator(), args, source->module_inputs());
    if(not source->get_debug_symbols().empty())
        m.add_debug_symbols(result, source->get_debug_symbols());
    replacements.emplace(source, result);
    resolution.memo.emplace(source, result);
    return result;
}

instruction_ref add_expression_values(
    module& m,
    const std::vector<sym::expr>& expressions,
    const std::unordered_map<sym::expr, instruction_ref>& root_sources,
    std::unordered_map<instruction_ref, instruction_ref>& replacements,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    auto sources = find_expression_sources(expressions, root_sources);
    if(not sources.has_value())
        MIGRAPHX_THROW("SPLIT_SYM_DIM: expression has no runtime shape source");
    if(sources->empty())
    {
        std::vector<int64_t> values;
        std::transform(
            expressions.begin(),
            expressions.end(),
            std::back_inserter(values),
            [](const auto& expression) { return static_cast<int64_t>(expression.eval_uint({})); });
        return m.add_literal(literal{shape{shape::int64_type, {values.size()}}, values});
    }

    replacement_resolution resolution;
    std::vector<instruction_ref> resolved_sources;
    std::transform(sources->begin(),
                   sources->end(),
                   std::back_inserter(resolved_sources),
                   [&](instruction_ref source) {
                       return resolve_replacement(
                           m, source, replacements, info_for_instruction, resolution);
                   });
    return m.add_instruction(
        make_op("eval_expr_from_shape", {{"expressions", to_value(expressions)}}),
        resolved_sources);
}

instruction_ref
find_output_value(const sliced_value& input,
                  const std::vector<std::pair<sliced_value, instruction_ref>>& output_values)
{
    auto found = std::find_if(output_values.begin(), output_values.end(), [&](const auto& value) {
        return value.first == input;
    });
    if(found == output_values.end())
    {
        auto same_source =
            std::count_if(output_values.begin(), output_values.end(), [&](const auto& value) {
                return value.first.source == input.source;
            });
        MIGRAPHX_THROW(
            "SPLIT_SYM_DIM: block output " + input.source->name() +
            " was not specialized before use; matching outputs=" + std::to_string(same_source));
    }
    return found->second;
}

void resolve_frame_inputs(
    module& m,
    block_frame& frame,
    std::size_t block_number,
    std::unordered_map<instruction_ref, instruction_ref>& replacements,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    const std::unordered_set<instruction_ref>& opaque_boundary_sources,
    const std::vector<std::pair<sliced_value, instruction_ref>>& output_values)
{
    replacement_resolution resolution;
    for(auto& input : frame.inputs)
    {
        auto source_info = info_for_instruction.find(input.clone_value.source);
        bool nonopaque_root_source =
            source_info != info_for_instruction.end() and source_info->second->root_source and
            not contains(opaque_boundary_sources, input.clone_value.source);
        input.select_input = input.clone_value.slice_axes.empty()
                                 ? resolve_replacement(m,
                                                       input.clone_value.source,
                                                       replacements,
                                                       info_for_instruction,
                                                       resolution,
                                                       block_number,
                                                       nonopaque_root_source)
                                 : find_output_value(input.clone_value, output_values);
    }
}

instruction_ref add_output_slice(
    module& m,
    const sliced_value& output,
    instruction_ref selected_output,
    const symbolic_op_info& info,
    const std::unordered_map<sym::expr, instruction_ref>& root_sources,
    std::unordered_map<instruction_ref, instruction_ref>& replacements,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    const auto& source_shape = output.source->get_shape();
    const auto& source_dims  = source_shape.dyn_dims();
    std::vector<int64_t> axes;
    std::vector<sym::expr> end_expressions;
    for(auto axis : info.output_symbolic_axes)
    {
        assert(axis < source_dims.size());
        if(contains(output.slice_axes, axis))
        {
            axes.push_back(static_cast<int64_t>(axis));
            end_expressions.push_back(source_dims.at(axis).sym_expr);
        }
    }
    std::vector<sym::expr> start_expressions(axes.size(), sym::lit(int64_t{0}));
    auto starts = add_expression_values(
        m, start_expressions, root_sources, replacements, info_for_instruction);
    auto ends =
        add_expression_values(m, end_expressions, root_sources, replacements, info_for_instruction);
    auto output_slice = make_op("dyn_slice",
                                {{"axes", axes},
                                 {"starts", to_value(start_expressions)},
                                 {"ends", to_value(end_expressions)},
                                 {"always_leq", true}});
    auto result       = m.add_instruction(output_slice, selected_output, starts, ends);
    result->set_normalized();
    if(not output.source->get_debug_symbols().empty())
        m.add_debug_symbols(result, output.source->get_debug_symbols());
    return result;
}

shape logical_output_shape(const sliced_value& output) { return output.source->get_shape(); }

void wire_select_module(
    module_pass_manager& mpm,
    const block_frame& frame,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    const std::unordered_map<sym::expr, instruction_ref>& root_sources,
    bool has_zero_specialization,
    std::vector<module> clones,
    const std::vector<clone_output_case>& clone_outputs,
    std::unordered_map<instruction_ref, instruction_ref>& replacements,
    std::vector<std::pair<sliced_value, instruction_ref>>& output_values)
{
    module& m = mpm.get_module();
    std::vector<module_ref> submodules;
    submodules.reserve(clones.size());
    for(auto& clone : clones)
    {
        auto name = clone.name();
        submodules.push_back(mpm.create_module(name, std::move(clone)));
    }

    std::vector<instruction_ref> selection_inputs;
    std::transform(frame.params.begin(),
                   frame.params.end(),
                   std::back_inserter(selection_inputs),
                   [&](const auto& input) { return frame.inputs.at(input.second).select_input; });
    std::vector<std::size_t> logical_output_indices;
    if(clone_outputs.empty())
    {
        std::vector<instruction_ref> output_sources;
        std::transform(frame.outputs.begin(),
                       frame.outputs.end(),
                       std::back_inserter(output_sources),
                       [](const auto& output) { return output.source; });
        logical_output_indices = deduplicate_block_outputs(output_sources).indices;
    }
    else
    {
        logical_output_indices = clone_outputs.front().output_indices;
        if(any_of(clone_outputs, [&](const auto& output) {
               return output.output_indices != logical_output_indices;
           }))
            MIGRAPHX_THROW("SPLIT_SYM_DIM: clone output aliases are inconsistent");
    }
    auto physical_output_count =
        *std::max_element(logical_output_indices.begin(), logical_output_indices.end()) + 1;
    auto physical_output_indices = range(physical_output_count);
    std::vector<std::size_t> representative_outputs;
    std::transform(physical_output_indices.begin(),
                   physical_output_indices.end(),
                   std::back_inserter(representative_outputs),
                   [&](std::size_t physical_output_index) {
                       auto found = std::find(logical_output_indices.begin(),
                                              logical_output_indices.end(),
                                              physical_output_index);
                       assert(found != logical_output_indices.end());
                       return static_cast<std::size_t>(
                           std::distance(logical_output_indices.begin(), found));
                   });

    std::vector<shape> body_output_shapes;
    std::vector<shape> logical_output_shapes;
    std::transform(representative_outputs.begin(),
                   representative_outputs.end(),
                   std::back_inserter(logical_output_shapes),
                   [&](std::size_t output_index) {
                       return logical_output_shape(frame.outputs.at(output_index));
                   });
    std::transform(
        physical_output_indices.begin(),
        physical_output_indices.end(),
        std::back_inserter(body_output_shapes),
        [&](std::size_t output_index) {
            if(clone_outputs.empty())
                return logical_output_shapes.at(output_index);
            auto source = frame.outputs.at(representative_outputs.at(output_index)).source;
            return dispatch_shape_for_clones(
                info_for_instruction.at(source)->dispatch_output, clone_outputs, output_index);
        });
    value op_value{{"output_dyn_shapes", to_value(shape{body_output_shapes})}};
    if(has_zero_specialization)
    {
        op_value["logical_output_dyn_shapes"] = to_value(shape{logical_output_shapes});
        op_value["num_inputs"]                = selection_inputs.size();
    }
    auto selection =
        m.add_instruction(make_op("select_module", op_value), selection_inputs, submodules);

    std::vector<instruction_ref> selected_outputs;
    std::transform(physical_output_indices.begin(),
                   physical_output_indices.end(),
                   std::back_inserter(selected_outputs),
                   [&](std::size_t output_index) {
                       return m.add_instruction(
                           make_op("get_tuple_elem", {{"index", output_index}}), selection);
                   });

    for(std::size_t output_index = 0; output_index < frame.outputs.size(); ++output_index)
    {
        const auto& output   = frame.outputs.at(output_index);
        auto selected_output = selected_outputs.at(logical_output_indices.at(output_index));
        auto sliced          = add_output_slice(m,
                                       output,
                                       selected_output,
                                       info_for_instruction.at(output.source)->info,
                                       root_sources,
                                       replacements,
                                       info_for_instruction);
        output_values.emplace_back(output, sliced);
        if(output == full_output_for(info_for_instruction, output.source))
            replacements.emplace(output.source, sliced);
    }
}

std::optional<sym::expr> find_root_runtime_extent(const root_spec& root, instruction_ref source)
{
    if(source->name() != "dyn_slice" or source->inputs().size() != 3 or
       not source->get_shape().symbolic() or not is_prefix_stable_dyn_slice(source))
        return std::nullopt;
    auto attributes = source->get_operator().to_value();
    if(not attributes.contains("axes"))
        return std::nullopt;
    auto axes = attributes.at("axes").to_vector<int64_t>();
    auto ends = source->inputs().at(2)->sym_eval();
    if(ends.empty() or ends.get().size() != axes.size())
        return std::nullopt;

    const auto& output = source->get_shape();
    auto indices       = range(axes.size());
    auto found         = std::find_if(indices.begin(), indices.end(), [&](auto i) {
        auto axis = normalize_axis(axes.at(i), output.ndim());
        return axis.has_value() and
               sym::same_symbol(output.dyn_dims().at(*axis).sym_expr, root.root);
    });
    if(found == indices.end())
        return std::nullopt;
    return ends.get()[*found];
}

bool specialization_is_reachable(
    const block_plan& block,
    const std::unordered_map<sym::expr, shape::dynamic_dimension::interval>& subranges,
    const std::unordered_map<sym::expr, instruction_ref>& root_sources)
{
    return all_of(block.roots, [&](const root_spec* root) {
        auto expression = find_root_runtime_extent(*root, root_sources.at(root->root));
        if(not expression.has_value())
            return true;
        std::unordered_map<sym::expr, sym::interval> bounds;
        for(const auto& variable : sym::find_variables(*expression))
        {
            auto found = std::find_if(subranges.begin(), subranges.end(), [&](const auto& entry) {
                return sym::same_symbol(entry.first, variable);
            });
            if(found != subranges.end())
                bounds.emplace(variable,
                               sym::interval{sym::pick_scalar::apply(found->second.min),
                                             sym::pick_scalar::apply(found->second.max)});
        }
        auto evaluated      = expression->eval_interval(bounds);
        auto selected       = subranges.at(root->root);
        sym::interval range = {sym::pick_scalar::apply(selected.min),
                               sym::pick_scalar::apply(selected.max)};
        return not(evaluated < range or evaluated > range);
    });
}

void specialize_blocks(
    module_pass_manager& mpm,
    const std::vector<block_plan>& blocks,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    const std::unordered_map<sym::expr, instruction_ref>& root_sources,
    const std::unordered_set<instruction_ref>& opaque_boundary_sources)
{
    // Replace each discovered block with one clone per cartesian product of its root buckets and a
    // select_module that dispatches to the compatible clone. The selected fixed-size outputs are
    // sliced back to their runtime extents before uses outside the block are rewired.
    module& m = mpm.get_module();
    std::unordered_map<instruction_ref, instruction_ref> replacements;
    std::vector<std::pair<sliced_value, instruction_ref>> output_values;
    auto original_outputs       = m.get_returns();
    auto required_block_outputs = find_required_block_outputs(blocks, info_for_instruction);

    auto block_numbers = range(blocks.size());
    for(auto&& [block_number, block] : views::zip(block_numbers, blocks))
    {
        auto frame = find_block_frame(
            m, block, info_for_instruction, block_number, root_sources, required_block_outputs);
        if(not frame.has_value())
            continue;
        resolve_frame_inputs(m,
                             *frame,
                             block_number,
                             replacements,
                             info_for_instruction,
                             opaque_boundary_sources,
                             output_values);

        std::unordered_map<sym::expr, sym::expr> fixed_substitutions;
        for(const auto* root : block.roots)
            if(root->interval.min == root->interval.max)
                fixed_substitutions.emplace(root->root, root->target_symbol);
        auto clone_count = std::accumulate(
            block.roots.begin(),
            block.roots.end(),
            std::size_t{1},
            [](auto count, const auto* root) { return count * root->specializations.size(); });
        std::vector<module> clones;
        std::vector<clone_output_case> clone_outputs;
        clones.reserve(clone_count);
        clone_outputs.reserve(clone_count);
        bool has_zero_specialization = false;
        for(std::size_t clone_index = 0; clone_index < clone_count; ++clone_index)
        {
            auto remaining = clone_index;
            std::unordered_map<sym::expr, std::size_t> freeze;
            std::unordered_map<sym::expr, shape::dynamic_dimension::interval> subranges;
            for(const auto* root : block.roots)
            {
                auto specialization_index = remaining % root->specializations.size();
                remaining /= root->specializations.size();
                auto specialization =
                    std::next(root->specializations.begin(), specialization_index);
                const auto& [target, runtime_range] = *specialization;
                freeze[root->root]                  = target;
                freeze[root->target_symbol]         = target;
                subranges[root->root]               = runtime_range;
            }
            assert(remaining == 0);
            if(not specialization_is_reachable(block, subranges, root_sources))
                continue;
            bool fixed_specialization =
                std::all_of(subranges.begin(), subranges.end(), [](const auto& entry) {
                    return entry.second.min == entry.second.max;
                });
            bool all_outputs_empty =
                fixed_specialization and
                std::all_of(frame->outputs.begin(), frame->outputs.end(), [&](const auto& output) {
                    auto s = logical_output_shape(output);
                    if(s.symbolic())
                        s = s.to_static(freeze);
                    return s.elements() == 0;
                });
            if(all_outputs_empty)
            {
                has_zero_specialization = true;
                continue;
            }
            auto name = m.name() + ":split_sym_dim_" + std::to_string(block_number) + "_" +
                        std::to_string(clone_index);
            auto built = build_clone(
                name, *frame, info_for_instruction, freeze, subranges, fixed_substitutions);
            clones.push_back(std::move(built.clone));
            clone_outputs.push_back(std::move(built.output_case));
        }
        wire_select_module(mpm,
                           *frame,
                           info_for_instruction,
                           root_sources,
                           has_zero_specialization,
                           std::move(clones),
                           clone_outputs,
                           replacements,
                           output_values);
    }
    std::vector<instruction_ref> outputs;
    std::transform(original_outputs.begin(),
                   original_outputs.end(),
                   std::back_inserter(outputs),
                   [&](instruction_ref output) {
                       replacement_resolution resolution;
                       return resolve_replacement(
                           m, output, replacements, info_for_instruction, resolution);
                   });
    m.replace_return(outputs);
    m.sort();
}

} // namespace

void split_sym_dim::apply(module_pass_manager& mpm) const
{
    module& m = mpm.get_module();

    auto registry = find_root_sources(m);
    if(registry.sources.empty())
        return;

    resolve_symbolic_dimensions_of_match resolve_symbolic_dimensions{.root_sources =
                                                                         registry.sources};
    match::find_matches(m, resolve_symbolic_dimensions);
    match::find_matches(m, find_eval_expr_from_intermediate_ins{.root_sources = registry.sources});
    run_passes(m, {dead_code_elimination{}});
    registry = find_root_sources(m);

    auto symbolic_instructions = find_all(iterator_for(m), [](instruction_ref ins) {
        if(ins->get_shape().symbolic())
            return true;
        return ins->name() == "topk" and not ins->inputs().empty() and
               ins->inputs().front()->get_shape().symbolic();
    });
    if(symbolic_instructions.empty())
        return;

    std::vector<symbolic_op_plan> infos;
    transform_if(
        symbolic_instructions.begin(),
        symbolic_instructions.end(),
        std::back_inserter(infos),
        [](instruction_ref ins) { return not starts_with(ins->name(), "@"); },
        [](instruction_ref ins) { return symbolic_op_plan{.info = analyze_instruction(ins)}; });
    std::unordered_map<instruction_ref, const symbolic_op_plan*> info_for_instruction;
    for(std::size_t rank : range(infos.size()))
    {
        auto& plan       = infos.at(rank);
        plan.rank        = rank;
        plan.root_source = contains(registry.boundaries, plan.info.ins);
        info_for_instruction.emplace(plan.info.ins, &plan);
    }
    remove_redundant_masks(infos, info_for_instruction);

    auto roots = collect_roots(m, registry);
    if(not roots.has_value())
        return;

    std::unordered_set<sym::expr> opaque_roots;
    for(const auto& [root, source] : registry.sources)
        if(contains(registry.dyn_concat_boundaries, source))
            opaque_roots.insert(root);
    auto pre_dyn_concat = find_pre_dyn_concat_analysis(registry, info_for_instruction);
    auto blocks         = discover_blocks(infos,
                                  *roots,
                                  info_for_instruction,
                                  max_clones,
                                  opaque_roots,
                                  pre_dyn_concat.instructions,
                                  pre_dyn_concat.data_roots);
    if(blocks.empty())
        return;

    order_blocks_for_staged_roots(blocks, registry, info_for_instruction);
    prepare_clone_infos(infos, info_for_instruction, *roots);

    std::unordered_set<instruction_ref> opaque_boundary_sources;
    std::transform(registry.dyn_concat_boundaries.begin(),
                   registry.dyn_concat_boundaries.end(),
                   std::inserter(opaque_boundary_sources, opaque_boundary_sources.end()),
                   [](const auto& entry) { return entry.first; });
    specialize_blocks(mpm, blocks, info_for_instruction, registry.sources, opaque_boundary_sources);
    run_passes(m, {dead_code_elimination{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
