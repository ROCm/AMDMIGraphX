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
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/eliminate_common_subexpression.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/insert_select_module.hpp>
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
#include <migraphx/utility_operators.hpp>
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

struct sliced_value : equality_comparable<sliced_value>
{
    instruction_ref source;
    std::vector<std::size_t> slice_axes;

    friend bool operator==(const sliced_value& x, const sliced_value& y)
    {
        return x.source == y.source and x.slice_axes == y.slice_axes;
    }
};

struct padding_use
{
    instruction_ref consumer;
    input_padding_requirement propagated;

    instruction_ref source() const
    {
        if(propagated.input >= consumer->inputs().size())
            MIGRAPHX_THROW("SPLIT_SYM_DIM: padding input is out of range");
        return consumer->inputs().at(propagated.input);
    }

    bool same_as(const padding_use& other) const
    {
        const auto& x = propagated.requirement;
        const auto& y = other.propagated.requirement;
        return consumer == other.consumer and propagated.input == other.propagated.input and
               x.axis == y.axis and sym::same_symbol(x.extent, y.extent) and x.fills == y.fills and
               x.semantics == y.semantics and x.materializable == y.materializable;
    }
};

struct padding_variant_step
{
    instruction_ref source;
    std::size_t input = 0;
    std::optional<float> boundary_pad_value;
};

struct padding_variant_plan
{
    std::vector<padding_variant_step> steps;

    bool
    same_suffix(std::size_t step, const padding_variant_plan& other, std::size_t other_step) const
    {
        if(step >= steps.size() or other_step >= other.steps.size())
            return false;
        if(steps.size() - step != other.steps.size() - other_step)
            return false;
        return std::equal(steps.begin() + step,
                          steps.end(),
                          other.steps.begin() + other_step,
                          [](const auto& left, const auto& right) {
                              return left.source == right.source and left.input == right.input and
                                     left.boundary_pad_value == right.boundary_pad_value;
                          });
    }
};

struct resolved_padding_variant_plan
{
    padding_requirement requirement;
    padding_variant_plan plan;
};

struct resolved_input_padding_plan
{
    std::optional<fill_kind> fill;
    std::optional<resolved_padding_variant_plan> variant;
    std::vector<std::size_t> repad_axes;
};

struct clone_input_plan
{
    sliced_value clone_value;
    std::optional<float> pad_value;
    std::vector<axis_mask> masks;
    std::optional<padding_variant_plan> padding_variant;
};

struct symbolic_op_plan
{
    symbolic_op_info info;
    std::size_t rank = 0;
    std::vector<clone_input_plan> clone_inputs;
    shape dispatch_output;

    bool is_shape_input(std::size_t index) const
    {
        return info.freezer and contains(info.shape_input_indices, index);
    }
};

struct symbolic_op_plans
{
    std::vector<symbolic_op_plan> values;
    std::unordered_map<instruction_ref, std::size_t> index_by_instruction;

    const symbolic_op_plan* find(instruction_ref ins) const
    {
        auto found = index_by_instruction.find(ins);
        if(found == index_by_instruction.end())
            return nullptr;
        return &values.at(found->second);
    }

    const symbolic_op_plan& at(instruction_ref ins) const
    {
        return values.at(index_by_instruction.at(ins));
    }

    bool contains(instruction_ref ins) const { return index_by_instruction.count(ins) > 0; }

    std::size_t size() const { return values.size(); }

    void build_index()
    {
        index_by_instruction.clear();
        index_by_instruction.reserve(values.size());
        for(auto rank : range(values.size()))
        {
            auto& plan = values.at(rank);
            plan.rank  = rank;
            if(not index_by_instruction.emplace(plan.info.ins, rank).second)
                MIGRAPHX_THROW("SPLIT_SYM_DIM: duplicate symbolic instruction plan");
        }
    }

    void remove_redundant_masks()
    {
        for(auto& plan : values)
        {
            auto& info       = plan.info;
            const auto& args = info.ins->inputs();
            for(auto&& [arg, operand] : views::zip(args, info.operands))
            {
                const auto* source = find(arg);
                if(source == nullptr)
                    continue;
                auto& masks = operand.masks;
                masks.erase(std::remove_if(masks.begin(),
                                           masks.end(),
                                           [&](const auto& mask) {
                                               return source->info.zeros_contracted_region(mask);
                                           }),
                            masks.end());
            }
        }
    }

    std::vector<sliced_value> clone_inputs_for(instruction_ref ins, bool planned) const
    {
        const auto* plan = find(ins);
        if(planned)
        {
            assert(plan != nullptr);
            std::vector<sliced_value> result;
            result.reserve(plan->clone_inputs.size());
            std::transform(plan->clone_inputs.begin(),
                           plan->clone_inputs.end(),
                           std::back_inserter(result),
                           [](const auto& input) { return input.clone_value; });
            return result;
        }
        std::vector<sliced_value> result;
        auto inputs        = ins->inputs();
        auto input_indices = range(inputs.size());
        for(auto&& [index, input] : views::zip(input_indices, inputs))
            if(plan == nullptr or not plan->is_shape_input(index))
                result.push_back({{}, input, {}});
        return result;
    }

    sliced_value full_output_for(instruction_ref source) const
    {
        sliced_value result{{}, source, {}};
        result.slice_axes = at(source).info.output_symbolic_axes;
        return result;
    }
};

struct block_assignment
{
    std::unordered_map<instruction_ref, std::size_t> indices;

    std::optional<std::size_t> find(instruction_ref ins) const
    {
        auto found = indices.find(ins);
        if(found == indices.end())
            return std::nullopt;
        return found->second;
    }

    bool is_assigned(instruction_ref ins) const { return indices.count(ins) > 0; }

    void insert(instruction_ref ins, std::size_t block)
    {
        if(not indices.emplace(ins, block).second)
            MIGRAPHX_THROW("SPLIT_SYM_DIM: instruction is assigned to multiple blocks");
    }

    void update(instruction_ref ins, std::size_t block) { indices.at(ins) = block; }

    void clear() { indices.clear(); }

    bool is_absorbable(instruction_ref ins,
                       const std::unordered_set<instruction_ref>& planned,
                       const symbolic_op_plans& plans) const
    {
        if(planned.count(ins) > 0)
            return true;
        if(is_assigned(ins))
            return false;
        if(starts_with(ins->name(), "@") or ins->name() == "eval_expr_from_shape" or
           not ins->module_inputs().empty())
            return false;
        const auto& s = ins->get_shape();
        if(not s.dynamic())
            return true;
        if(not s.symbolic())
            return false;
        if(not s.has_fixed_strides())
            return false;
        const auto* plan = plans.find(ins);
        if(plan == nullptr)
            return false;
        const auto& info = plan->info;
        if(not info.supported or (not info.freezer and not s.is_fixed()))
            return false;
        const auto& inputs = ins->inputs();
        assert(inputs.size() == info.operands.size());
        return not any_of(
            views::zip(info.operands, inputs), unpack([&](const auto& operand, auto input) {
                return operand.pad_value.has_value() and is_absorbable(input, planned, plans);
            }));
    }

    void append_connection(std::set<std::pair<std::size_t, std::size_t>>& connections,
                           instruction_ref consumer,
                           std::size_t input,
                           const symbolic_op_plans& plans) const
    {
        const auto* consumer_plan = plans.find(consumer);
        auto consumer_block       = find(consumer);
        if(consumer_plan == nullptr or not consumer_block.has_value())
            return;
        auto producer             = consumer->inputs().at(input);
        const auto* producer_plan = plans.find(producer);
        auto producer_block       = find(producer);
        if(producer_plan == nullptr or not producer_block.has_value() or
           producer_block == consumer_block)
            return;
        connections.insert({consumer_plan->rank, input});
    }
};

struct dependency_cache
{
    const block_assignment& assignment;
    const symbolic_op_plans& plans;
    std::unordered_map<instruction_ref, std::set<std::size_t>> values;

    void discover(instruction_ref source)
    {
        if(contains(values, source))
            return;

        std::set<std::size_t> dependencies;
        const auto* source_plan = plans.find(source);
        auto source_block       = assignment.find(source);
        if(source_block.has_value())
        {
            dependencies.insert(*source_block);
        }
        else
        {
            const auto& inputs = source->inputs();
            auto input_indices = range(inputs.size());
            std::for_each(input_indices.begin(), input_indices.end(), [&](auto input_index) {
                if(source_plan != nullptr and source_plan->info.freezer and
                   contains(source_plan->info.shape_input_indices, input_index))
                    return;
                auto input = inputs.at(input_index);
                discover(input);
                const auto& found = at(input);
                dependencies.insert(found.begin(), found.end());
            });
        }
        values.emplace(source, std::move(dependencies));
    }

    const std::set<std::size_t>& at(instruction_ref source) const { return values.at(source); }
};

bool has_symbolic_param(const module& m)
{
    auto param_shapes = m.get_parameter_shapes();
    return any_of(param_shapes, [](const auto& p) { return p.second.symbolic(); });
}

std::unordered_map<sym::expr, instruction_ref> find_root_sources(const module& m)
{
    std::unordered_map<sym::expr, instruction_ref> result;
    for(const auto& name : m.get_parameter_names())
    {
        auto parameter = m.get_parameter(name);
        const auto& s  = parameter->get_shape();
        if(not s.symbolic())
            continue;
        for(const auto& d : s.dyn_dims())
            if(d.is_symbolic() and d.sym_expr.name() == "variable")
                result.emplace(sym::as_symbol(d.sym_expr), parameter);
    }
    return result;
}

std::optional<std::vector<instruction_ref>>
find_expression_sources(const std::vector<sym::expr>& expressions,
                        const std::unordered_map<sym::expr, instruction_ref>& root_sources,
                        const std::vector<instruction_ref>& sources)
{
    std::unordered_set<instruction_ref> required_sources;
    if(any_of(expressions, [&](const auto& expression) {
           auto variables = sym::find_variables(expression);
           return any_of(variables, [&](const auto& variable) {
               if(not contains(root_sources, variable))
                   return true;
               required_sources.insert(root_sources.at(variable));
               return false;
           });
       }))
        return std::nullopt;

    std::vector<instruction_ref> result;
    std::copy_if(sources.begin(),
                 sources.end(),
                 std::back_inserter(result),
                 [&](instruction_ref source) { return contains(required_sources, source); });
    return result;
}

struct resolve_symbolic_dimensions_of_match : match::supports_dynamic_shapes
{
    std::unordered_map<sym::expr, instruction_ref> root_sources;
    std::vector<instruction_ref> sources;

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
        const auto expressions  = symbolic_value.get().to_vector();
        auto expression_sources = find_expression_sources(expressions, root_sources, sources);
        if(not expression_sources.has_value())
            return;
        if(expression_sources->empty())
            expression_sources = sources;
        m.replace_instruction(
            ins,
            make_op("eval_expr_from_shape", {{"expressions", to_value(expressions)}}),
            *expression_sources);
    }
};

struct find_eval_expr_from_intermediate_ins : match::supports_dynamic_shapes
{
    std::unordered_map<sym::expr, instruction_ref> root_sources;
    std::vector<instruction_ref> sources;

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
        auto expression_sources = find_expression_sources(expressions, root_sources, sources);
        if(not expression_sources.has_value() or expression_sources->empty())
            return;
        m.replace_instruction(
            ins,
            make_op("eval_expr_from_shape", {{"expressions", to_value(expressions)}}),
            *expression_sources);
    }
};

struct root_spec
{
    sym::expr root;
    std::string name;
    shape::dynamic_dimension::interval interval;
    sym::expr target_symbol;
    std::map<std::size_t, shape::dynamic_dimension::interval> specializations;
};

bool has_bounded_interval(const sym::expr& expression)
{
    const auto interval = expression.eval_interval_default();
    return sym::to<int64_t>(interval.min) != std::numeric_limits<int64_t>::lowest() and
           sym::to<int64_t>(interval.max) != std::numeric_limits<int64_t>::max();
}

struct root_collection
{
    std::optional<std::unordered_map<sym::expr, root_spec>> specs =
        std::unordered_map<sym::expr, root_spec>{};

    void append_direct(const shape::dynamic_dimension& dimension)
    {
        if(not specs.has_value() or not dimension.is_symbolic())
            return;
        if(dimension.sym_expr.name() != "variable")
            return;
        if(not has_bounded_interval(dimension.sym_expr))
        {
            specs.reset();
            return;
        }

        auto root        = sym::as_symbol(dimension.sym_expr);
        auto name        = root.to_string();
        auto interval    = dimension.get_interval();
        auto optimal_set = dimension.get_optimals();
        if(any_of(optimal_set, [&](auto x) { return x < interval.min or x > interval.max; }))
        {
            specs.reset();
            return;
        }
        optimal_set.insert(interval.min);
        optimal_set.insert(interval.max);
        std::map<std::size_t, shape::dynamic_dimension::interval> specializations;
        auto lower = interval.min;
        for(auto optimal : optimal_set)
        {
            specializations.emplace(optimal, shape::dynamic_dimension::interval{lower, optimal});
            lower = optimal + 1;
        }

        auto found = specs->find(root);
        if(found != specs->end())
        {
            if(found->second.interval != interval or
               found->second.specializations != specializations)
                specs.reset();
            return;
        }

        specs->emplace(root,
                       root_spec{root, std::move(name), interval, {}, std::move(specializations)});
    }

    bool valid() const { return specs.has_value(); }

    bool contains_roots_for(const shape::dynamic_dimension& dimension) const
    {
        if(not is_variable_axis(dimension))
            return true;
        if(not specs.has_value() or not has_bounded_interval(dimension.sym_expr))
            return false;
        auto variables = sym::find_variables(dimension.sym_expr);
        return not variables.empty() and
               all_of(variables, [&](const auto& variable) { return contains(*specs, variable); });
    }
};

// Collect the independent symbols that enter through module parameters. Each symbol's min,
// optimals, and max partition its runtime interval into specialization buckets.
std::optional<std::vector<root_spec>> collect_roots(const module& m)
{
    root_collection collected;
    auto parameters = m.get_parameters();
    for(auto parameter : parameters)
    {
        const auto& s = parameter->get_shape();
        if(s.dynamic() and not s.symbolic())
            return std::nullopt;
        if(not s.symbolic())
            continue;
        for(const auto& dimension : s.dyn_dims())
            collected.append_direct(dimension);
        if(not collected.valid())
            return std::nullopt;
    }
    auto& root_specs = *collected.specs;
    if(root_specs.empty())
        return std::nullopt;
    if(not all_of(parameters, [&](auto parameter) {
           const auto& s = parameter->get_shape();
           return not s.symbolic() or all_of(s.dyn_dims(), [&](const auto& dimension) {
               return collected.contains_roots_for(dimension);
           });
       }))
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
    std::transform(roots.begin(),
                   roots.end(),
                   std::inserter(symbol_names, symbol_names.end()),
                   [](const auto& root) { return root.name; });

    for(auto& root : roots)
    {
        std::string target_name = "split_sym_dim_" + root.name + "_target";
        while(contains(symbol_names, target_name))
            target_name += "_";
        symbol_names.insert(target_name);
        std::set<sym::scalar> targets;
        std::transform(
            root.specializations.begin(),
            root.specializations.end(),
            std::inserter(targets, targets.end()),
            [](const auto& specialization) { return sym::scalar{specialization.first}; });
        if(root.interval.min == root.interval.max)
            root.target_symbol = sym::lit(root.interval.min);
        else
            root.target_symbol = sym::var(
                std::move(target_name), {root.interval.min, root.interval.max}, std::move(targets));
    }
    return roots;
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

void append_shape_roots(std::unordered_set<sym::expr>& result, const shape& s)
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

std::optional<std::vector<const root_spec*>> find_instruction_roots(
    const symbolic_op_info& info, const std::vector<root_spec>& roots, std::size_t max_clones)
{
    std::unordered_set<sym::expr> required;
    append_shape_roots(required, info.get_output_shape());
    for(const auto& input : info.get_input_shapes())
        append_shape_roots(required, input);
    for(const auto& operand : info.operands)
    {
        for(const auto& mask : operand.masks)
        {
            auto variables = sym::find_variables(mask.extent);
            required.insert(variables.begin(), variables.end());
        }
    }

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

std::optional<std::set<fill_kind>> intersect_fills(const std::set<fill_kind>& x,
                                                   const std::set<fill_kind>& y)
{
    std::set<fill_kind> result;
    std::set_intersection(
        x.begin(), x.end(), y.begin(), y.end(), std::inserter(result, result.end()));
    if(result.empty())
        return std::nullopt;
    return result;
}

std::optional<padding_requirement> meet_padding_requirements(const padding_requirement& x,
                                                             const padding_requirement& y)
{
    if(x.axis != y.axis or not sym::same_symbol(x.extent, y.extent))
        return std::nullopt;
    auto fills = intersect_fills(x.fills, y.fills);
    if(not fills.has_value())
        return std::nullopt;
    auto materializable = x.materializable and y.materializable;
    if(x.semantics == y.semantics)
        return padding_requirement{
            x.axis, x.extent, std::move(*fills), x.semantics, materializable};
    if(x.semantics == padding_semantics::exact or y.semantics == padding_semantics::exact)
        return padding_requirement{
            x.axis, x.extent, std::move(*fills), padding_semantics::exact, materializable};
    return std::nullopt;
}

std::optional<input_padding_requirement>
resolve_boundary_requirement(const symbolic_op_plan* consumer,
                             const input_padding_requirement& propagated)
{
    const auto& info = consumer->info;
    if(propagated.input >= info.operands.size() or propagated.requirement.fills.empty() or
       not propagated.requirement.materializable)
        return std::nullopt;
    const auto& operand = info.operands.at(propagated.input);
    const auto& input   = info.ins->inputs().at(propagated.input)->get_shape();
    if(not operand.pad_value.has_value() or propagated.requirement.axis >= input.ndim() or
       not all_of(propagated.requirement.fills,
                  [&](auto fill) { return supports_fill(input.type(), fill); }))
        return std::nullopt;

    auto result = propagated;
    for(const auto& existing : operand.padding_requirements)
    {
        auto fills = intersect_fills(result.requirement.fills, existing.fills);
        if(not fills.has_value())
            return std::nullopt;
        result.requirement.fills = std::move(*fills);
    }
    return result;
}

std::optional<fill_kind> choose_fill(const std::set<fill_kind>& fills)
{
    if(fills.empty())
        return std::nullopt;
    return *fills.begin();
}

struct padding_resolution
{
    const symbolic_op_plans& plans;
    std::vector<const symbolic_op_plan*> ops;
    std::vector<padding_use> requirements;
    std::unordered_map<instruction_ref, std::vector<resolved_input_padding_plan>> resolved_inputs;
    std::unordered_set<instruction_ref> included;
    std::unordered_map<instruction_ref, std::vector<padding_use>> propagated;
    bool valid = true;

    void reject() { valid = false; }

    bool has_requirement(instruction_ref consumer,
                         std::size_t input,
                         const padding_requirement& requirement) const
    {
        return any_of(requirements, [&](const auto& existing) {
            const auto& value = existing.propagated.requirement;
            return existing.consumer == consumer and existing.propagated.input == input and
                   value.axis == requirement.axis and
                   sym::same_symbol(value.extent, requirement.extent) and
                   value.fills == requirement.fills and value.semantics == requirement.semantics and
                   value.materializable == requirement.materializable;
        });
    }

    void collect_internal_requirements()
    {
        for(const auto* plan : ops)
        {
            const auto& info   = plan->info;
            const auto& inputs = info.ins->inputs();
            for(auto input_index : range(inputs.size()))
            {
                if(not contains(included, inputs.at(input_index)))
                    continue;
                const auto& operand = info.operands.at(input_index);
                if(not operand.pad_value.has_value() or operand.retained_slice_axes.empty())
                    continue;
                auto matching =
                    find_all(operand.padding_requirements, [&](const auto& requirement) {
                        return contains(operand.retained_slice_axes, requirement.axis);
                    });
                if(matching.size() != operand.retained_slice_axes.size())
                {
                    reject();
                    return;
                }
                for(const auto& requirement : matching)
                    if(not has_requirement(info.ins, input_index, requirement))
                        requirements.push_back(
                            {info.ins, {static_cast<std::size_t>(input_index), requirement}});
            }
        }
    }

    void add_propagated(padding_use use)
    {
        auto& uses = propagated[use.source()];
        if(none_of(uses, [&](const auto& existing) { return existing.same_as(use); }))
            uses.push_back(std::move(use));
    }

    std::vector<const padding_use*> find_axis_group(const std::vector<padding_use>& uses,
                                                    const padding_use& representative) const
    {
        std::vector<const padding_use*> result;
        const auto& grouped = representative.propagated.requirement;
        transform_if(
            uses.begin(),
            uses.end(),
            std::back_inserter(result),
            [&](const auto& use) {
                const auto& value = use.propagated.requirement;
                return value.axis == grouped.axis and
                       sym::same_symbol(value.extent, grouped.extent);
            },
            [](const auto& use) { return &use; });
        return result;
    }

    std::vector<const padding_use*> find_input_group(const std::vector<padding_use>& uses,
                                                     const padding_use& representative) const
    {
        std::vector<const padding_use*> result;
        transform_if(
            uses.begin(),
            uses.end(),
            std::back_inserter(result),
            [&](const auto& use) {
                return use.consumer == representative.consumer and
                       use.propagated.input == representative.propagated.input;
            },
            [](const auto& use) { return &use; });
        return result;
    }

    std::vector<const padding_use*>
    find_axis_representatives(const std::vector<padding_use>& uses) const
    {
        std::vector<const padding_use*> result;
        transform_if(
            uses.begin(),
            uses.end(),
            std::back_inserter(result),
            [&](const auto& use) {
                return none_of(result, [&](const auto* existing) {
                    const auto& x = existing->propagated.requirement;
                    const auto& y = use.propagated.requirement;
                    return x.axis == y.axis and sym::same_symbol(x.extent, y.extent);
                });
            },
            [](const auto& use) { return &use; });
        return result;
    }

    std::vector<const padding_use*>
    find_input_representatives(const std::vector<padding_use>& uses) const
    {
        std::vector<const padding_use*> result;
        transform_if(
            uses.begin(),
            uses.end(),
            std::back_inserter(result),
            [&](const auto& use) {
                return none_of(result, [&](const auto* existing) {
                    return existing->consumer == use.consumer and
                           existing->propagated.input == use.propagated.input;
                });
            },
            [](const auto& use) { return &use; });
        return result;
    }

    std::optional<padding_requirement> meet_group(const std::vector<const padding_use*>& uses) const
    {
        if(uses.empty())
            return std::nullopt;
        std::optional<padding_requirement> result = uses.front()->propagated.requirement;
        for(const auto* use : uses)
        {
            result = meet_padding_requirements(*result, use->propagated.requirement);
            if(not result.has_value())
                return std::nullopt;
        }
        return result;
    }

    std::optional<resolved_padding_variant_plan> find_variant(instruction_ref source,
                                                              const padding_use& use) const
    {
        if(use.source() != source)
            return std::nullopt;
        resolved_padding_variant_plan result{use.propagated.requirement, {}};
        auto current_source      = source;
        auto current_requirement = use.propagated.requirement;
        while(true)
        {
            const auto* plan = plans.find(current_source);
            if(plan == nullptr or current_requirement.semantics != padding_semantics::exact or
               not plan->info.propagate_padding)
                return std::nullopt;
            auto next = plan->info.propagate_padding(current_source, current_requirement);
            if(not next.has_value() or next->size() != 1)
                return std::nullopt;
            const auto& input = next->front();
            if(input.input >= current_source->inputs().size() or
               input.requirement.semantics != padding_semantics::exact or
               input.requirement.fills != current_requirement.fills)
                return std::nullopt;

            padding_variant_step step{current_source, input.input, std::nullopt};
            auto input_source = current_source->inputs().at(input.input);
            if(not contains(included, input_source))
            {
                auto boundary = resolve_boundary_requirement(plan, input);
                if(not boundary.has_value())
                    return std::nullopt;
                auto fill = choose_fill(boundary->requirement.fills);
                if(not fill.has_value())
                    return std::nullopt;
                step.boundary_pad_value = fill_value(*fill);
                result.plan.steps.push_back(step);
                return result;
            }
            result.plan.steps.push_back(step);
            current_source      = input_source;
            current_requirement = input.requirement;
        }
    }

    void set_variant(instruction_ref source, const padding_use& use)
    {
        auto found = find_variant(source, use);
        if(not found.has_value())
        {
            reject();
            return;
        }
        auto& inputs  = resolved_inputs
                            .try_emplace(use.consumer,
                                         std::vector<resolved_input_padding_plan>(
                                             use.consumer->inputs().size()))
                            .first->second;
        auto& variant = inputs.at(use.propagated.input).variant;
        if(variant.has_value())
        {
            const auto& x = variant->requirement;
            const auto& y = use.propagated.requirement;
            if(x.axis != y.axis or not sym::same_symbol(x.extent, y.extent) or x.fills != y.fills or
               x.semantics != y.semantics or x.materializable != y.materializable)
                reject();
            return;
        }
        variant = std::move(*found);
    }

    void propagate_group(instruction_ref source, const std::vector<const padding_use*>& uses)
    {
        auto merged = meet_group(uses);
        if(not merged.has_value())
        {
            for(const auto* use : uses)
            {
                set_variant(source, *use);
                if(not valid)
                    return;
            }
            return;
        }

        const auto* plan = plans.find(source);
        if(plan == nullptr or not plan->info.propagate_padding)
        {
            reject();
            return;
        }
        auto inputs = plan->info.propagate_padding(source, *merged);
        if(not inputs.has_value())
        {
            reject();
            return;
        }
        for(const auto& input : *inputs)
        {
            if(input.input >= source->inputs().size())
            {
                reject();
                return;
            }
            add_propagated({source, input});
        }
    }

    void propagate_internal()
    {
        std::unordered_map<instruction_ref, std::size_t> remaining_consumers;
        for(const auto* plan : ops)
            remaining_consumers.emplace(plan->info.ins, 0);
        for(const auto* plan : ops)
            for(auto source : plan->info.ins->inputs())
                if(contains(included, source))
                    remaining_consumers.at(source)++;

        std::vector<const symbolic_op_plan*> ready;
        std::copy_if(ops.begin(), ops.end(), std::back_inserter(ready), [&](const auto* plan) {
            return remaining_consumers.at(plan->info.ins) == 0;
        });
        std::size_t processed = 0;
        while(not ready.empty() and valid)
        {
            auto next =
                std::max_element(ready.begin(), ready.end(), [](const auto* x, const auto* y) {
                    return std::less<std::size_t>{}(x->rank, y->rank);
                });
            const auto* plan = *next;
            ready.erase(next);
            processed++;
            auto found = propagated.find(plan->info.ins);
            if(found != propagated.end())
            {
                auto representatives = find_axis_representatives(found->second);
                for(const auto* representative : representatives)
                {
                    propagate_group(plan->info.ins,
                                    find_axis_group(found->second, *representative));
                    if(not valid)
                        return;
                }
            }
            for(auto source : plan->info.ins->inputs())
            {
                if(not contains(included, source))
                    continue;
                auto& count = remaining_consumers.at(source);
                count--;
                if(count == 0)
                    ready.push_back(&plans.at(source));
            }
        }
        if(processed != ops.size())
            reject();
    }

    void resolve_boundaries()
    {
        for(const auto& [source, uses] : propagated)
        {
            if(contains(included, source))
                continue;
            auto representatives = find_input_representatives(uses);
            for(const auto* representative : representatives)
            {
                auto group = find_input_group(uses, *representative);
                std::optional<std::set<fill_kind>> fills;
                std::vector<std::size_t> repad_axes;
                for(const auto* use : group)
                {
                    const auto* consumer = plans.find(use->consumer);
                    if(consumer == nullptr)
                    {
                        reject();
                        return;
                    }
                    auto boundary = resolve_boundary_requirement(consumer, use->propagated);
                    if(not boundary.has_value())
                    {
                        reject();
                        return;
                    }
                    fills = fills.has_value()
                                ? intersect_fills(*fills, boundary->requirement.fills)
                                : std::optional<std::set<fill_kind>>{boundary->requirement.fills};
                    if(not fills.has_value())
                    {
                        reject();
                        return;
                    }
                    if(not contains(repad_axes, boundary->requirement.axis))
                        repad_axes.push_back(boundary->requirement.axis);
                }
                auto fill = choose_fill(*fills);
                if(not fill.has_value())
                {
                    reject();
                    return;
                }
                auto& resolved = resolved_inputs
                                     .try_emplace(representative->consumer,
                                                  std::vector<resolved_input_padding_plan>(
                                                      representative->consumer->inputs().size()))
                                     .first->second.at(representative->propagated.input);
                resolved.fill  = fill;
                resolved.repad_axes = std::move(repad_axes);
            }
        }
    }

    void resolve()
    {
        std::transform(ops.begin(),
                       ops.end(),
                       std::inserter(included, included.end()),
                       [](const auto* plan) { return plan->info.ins; });
        collect_internal_requirements();
        if(not valid)
            return;
        resolved_inputs.clear();
        propagated.clear();
        for(const auto& requirement : requirements)
            add_propagated(requirement);
        propagate_internal();
        if(valid)
            resolve_boundaries();
    }
};

struct block_plan
{
    std::vector<const symbolic_op_plan*> ops;
    std::vector<const root_spec*> roots;
    std::vector<padding_use> padding_requirements;
    std::unordered_map<instruction_ref, std::vector<resolved_input_padding_plan>> resolved_inputs;

    std::unordered_set<instruction_ref> instructions() const
    {
        std::unordered_set<instruction_ref> result;
        std::transform(ops.begin(),
                       ops.end(),
                       std::inserter(result, result.end()),
                       [](const auto* plan) { return plan->info.ins; });
        return result;
    }

    std::size_t rank() const
    {
        assert(not ops.empty());
        auto result = std::min_element(
            ops.begin(), ops.end(), [](const symbolic_op_plan* x, const symbolic_op_plan* y) {
                return x->rank < y->rank;
            });
        return (*result)->rank;
    }

    bool is_closed(const symbolic_op_plans& plans, const block_assignment& assignment) const
    {
        auto included = instructions();

        std::unordered_set<instruction_ref> visited;
        std::unordered_set<instruction_ref> boundary_visited;
        auto boundary_reaches_block  = fix<bool>([&](auto self, instruction_ref dependency) {
            if(not boundary_visited.insert(dependency).second)
                return false;
            if(contains(included, dependency))
                return true;
            return any_of(dependency->inputs(), self);
        });
        auto dependencies_are_closed = fix<bool>([&](auto self, instruction_ref current) {
            if(not visited.insert(current).second)
                return true;
            if(contains(included, current))
                return true;
            if(not assignment.is_absorbable(current, included, plans))
                return not boundary_reaches_block(current);
            return all_of(current->inputs(), self);
        });

        for(const auto* op : ops)
        {
            const auto& info = op->info;
            const auto& args = info.ins->inputs();
            assert(args.size() == info.operands.size());
            for(std::size_t index = 0; index < args.size(); ++index)
            {
                if(contains(info.shape_input_indices, index))
                    continue;
                auto source = args.at(index);
                if(contains(included, source))
                    continue;
                if(not dependencies_are_closed(source))
                    return false;
            }
        }
        return true;
    }

    std::optional<std::vector<const root_spec*>> merge_roots(const block_plan& source,
                                                             std::size_t max_clones) const
    {
        std::vector<const root_spec*> result;
        result.reserve(roots.size() + source.roots.size());
        std::set_union(roots.begin(),
                       roots.end(),
                       source.roots.begin(),
                       source.roots.end(),
                       std::back_inserter(result),
                       [](const auto* x, const auto* y) { return x->name < y->name; });
        if(not roots_fit_clone_limit(result, max_clones))
            return std::nullopt;
        return result;
    }

    std::optional<block_plan> resolved(const symbolic_op_plans& plans,
                                       const block_assignment& assignment) const
    {
        if(not is_closed(plans, assignment))
            return std::nullopt;
        padding_resolution resolution{
            .plans = plans, .ops = ops, .requirements = padding_requirements};
        resolution.resolve();
        if(not resolution.valid)
            return std::nullopt;
        auto result                 = *this;
        result.padding_requirements = std::move(resolution.requirements);
        result.resolved_inputs      = std::move(resolution.resolved_inputs);
        return result;
    }

    std::optional<block_plan> merged_with(const block_plan& source,
                                          const symbolic_op_plans& plans,
                                          const block_assignment& assignment,
                                          std::size_t max_clones) const
    {
        auto merged_roots = merge_roots(source, max_clones);
        if(not merged_roots.has_value())
            return std::nullopt;
        block_plan result;
        result.ops = ops;
        result.ops.insert(result.ops.end(), source.ops.begin(), source.ops.end());
        result.roots                = std::move(*merged_roots);
        result.padding_requirements = padding_requirements;
        for(const auto& requirement : source.padding_requirements)
            if(none_of(result.padding_requirements,
                       [&](const auto& existing) { return existing.same_as(requirement); }))
                result.padding_requirements.push_back(requirement);
        return result.resolved(plans, assignment);
    }

    std::set<std::pair<std::size_t, std::size_t>>
    find_connections(const symbolic_op_plans& plans, const block_assignment& assignment) const
    {
        std::set<std::pair<std::size_t, std::size_t>> result;
        std::for_each(ops.begin(), ops.end(), [&](const auto* plan) {
            const auto& info   = plan->info;
            auto input_indices = range(info.ins->inputs().size());
            std::for_each(input_indices.begin(), input_indices.end(), [&](auto input) {
                assignment.append_connection(result, info.ins, input, plans);
            });
            std::for_each(info.ins->outputs().begin(), info.ins->outputs().end(), [&](auto output) {
                const auto& inputs = output->inputs();
                auto output_inputs = range(inputs.size());
                std::for_each(output_inputs.begin(), output_inputs.end(), [&](auto input) {
                    if(inputs.at(input) == info.ins)
                        assignment.append_connection(result, output, input, plans);
                });
            });
        });
        return result;
    }
};

struct block_merge
{
    std::size_t target;
    std::size_t source;
    block_plan block;
    std::vector<std::set<std::size_t>> dependencies;
};

bool is_data_input(instruction_ref source, instruction_ref consumer, const symbolic_op_plans& plans)
{
    const auto* plan = plans.find(consumer);
    auto indices     = range(consumer->inputs().size());
    return any_of(indices, [&](auto index) {
        return consumer->inputs().at(index) == source and
               (plan == nullptr or not contains(plan->info.shape_input_indices, index));
    });
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

struct block_dependency_graph
{
    std::vector<block_plan> blocks;
    std::vector<std::set<std::size_t>> dependencies;
    block_assignment assignment;

    std::optional<std::size_t> find_block(instruction_ref ins) const
    {
        return assignment.find(ins);
    }

    bool is_planned(instruction_ref ins) const { return assignment.is_assigned(ins); }

    void add(block_plan block)
    {
        auto block_index = blocks.size();
        for(const auto* plan : block.ops)
            assignment.insert(plan->info.ins, block_index);
        blocks.push_back(std::move(block));
        dependencies.emplace_back();
    }

    std::set<std::pair<std::size_t, std::size_t>>
    find_all_connections(const symbolic_op_plans& plans) const
    {
        std::set<std::pair<std::size_t, std::size_t>> result;
        std::for_each(plans.values.begin(), plans.values.end(), [&](const auto& plan) {
            const auto& info   = plan.info;
            auto input_indices = range(info.ins->inputs().size());
            std::for_each(input_indices.begin(), input_indices.end(), [&](auto input) {
                assignment.append_connection(result, info.ins, input, plans);
            });
        });
        return result;
    }

    void update_dependencies(const symbolic_op_plans& plans)
    {
        dependency_cache cache{.assignment = assignment, .plans = plans};
        cache.values.reserve(plans.size());
        dependencies.clear();
        dependencies.reserve(blocks.size());
        std::transform(
            blocks.begin(), blocks.end(), std::back_inserter(dependencies), [&](const auto& block) {
                std::set<std::size_t> block_dependencies;
                if(block.ops.empty())
                    return block_dependencies;
                std::for_each(block.ops.begin(), block.ops.end(), [&](const auto* plan) {
                    const auto& info   = plan->info;
                    const auto& inputs = info.ins->inputs();
                    auto input_indices = range(inputs.size());
                    std::for_each(
                        input_indices.begin(), input_indices.end(), [&](auto input_index) {
                            if(contains(info.shape_input_indices, input_index))
                                return;
                            auto input = inputs.at(input_index);
                            cache.discover(input);
                            const auto& found = cache.at(input);
                            block_dependencies.insert(found.begin(), found.end());
                        });
                });
                auto block_index = find_block(block.ops.front()->info.ins);
                assert(block_index.has_value());
                block_dependencies.erase(*block_index);
                return block_dependencies;
            });
    }

    std::optional<std::vector<std::size_t>> order() const
    {
        assert(blocks.size() == dependencies.size());
        std::vector<std::pair<std::size_t, std::size_t>> ranked_blocks;
        auto block_indices = range(blocks.size());
        transform_if(
            block_indices.begin(),
            block_indices.end(),
            std::back_inserter(ranked_blocks),
            [&](auto block_index) { return not blocks.at(block_index).ops.empty(); },
            [&](auto block_index) {
                return std::make_pair(blocks.at(block_index).rank(), block_index);
            });
        std::sort(ranked_blocks.begin(), ranked_blocks.end());

        enum class visit_state
        {
            unvisited,
            visiting,
            complete
        };
        std::vector<visit_state> states(blocks.size(), visit_state::unvisited);
        std::vector<std::size_t> result;
        result.reserve(ranked_blocks.size());
        auto append_block = fix<bool>([&](auto self, std::size_t block_index) {
            auto& state = states.at(block_index);
            if(state == visit_state::complete)
                return true;
            if(state == visit_state::visiting)
                return false;
            assert(not blocks.at(block_index).ops.empty());
            state = visit_state::visiting;
            if(not all_of(dependencies.at(block_index), self))
                return false;
            state = visit_state::complete;
            result.push_back(block_index);
            return true;
        });
        if(not all_of(ranked_blocks,
                      [&](const auto& ranked_block) { return append_block(ranked_block.second); }))
            return std::nullopt;
        return result;
    }

    void contract(std::size_t target, std::size_t source)
    {
        assert(target < dependencies.size());
        assert(source < dependencies.size());
        std::vector<std::set<std::size_t>> result(dependencies.size());
        auto consumers = range(dependencies.size());
        std::for_each(consumers.begin(), consumers.end(), [&](auto consumer) {
            auto merged_consumer        = consumer == source ? target : consumer;
            const auto& producer_blocks = dependencies.at(consumer);
            std::for_each(producer_blocks.begin(), producer_blocks.end(), [&](auto dependency) {
                auto merged_dependency = dependency == source ? target : dependency;
                if(merged_consumer != merged_dependency)
                    result.at(merged_consumer).insert(merged_dependency);
            });
        });
        dependencies = std::move(result);
    }

    // Coalescing turns per-instruction specialization candidates into larger regions so connected
    // operations share one select_module. A merge is accepted only when the combined region is
    // closed over its dynamic dependencies, the cartesian product of root buckets stays within
    // max_clones, and contracting the two blocks keeps the dependency graph acyclic.
    std::optional<block_merge> find_merge(std::size_t x,
                                          std::size_t y,
                                          const symbolic_op_plans& plans,
                                          std::size_t max_clones) const
    {
        const auto target = std::min(x, y);
        const auto source = std::max(x, y);
        if(target == source or blocks.at(target).ops.empty() or blocks.at(source).ops.empty())
            return std::nullopt;

        auto merged_block =
            blocks.at(target).merged_with(blocks.at(source), plans, assignment, max_clones);
        if(not merged_block.has_value())
            return std::nullopt;
        auto candidate              = *this;
        candidate.blocks.at(target) = *merged_block;
        candidate.blocks.at(source).ops.clear();
        candidate.contract(target, source);
        if(not candidate.order().has_value())
            return std::nullopt;
        return block_merge{
            target, source, std::move(*merged_block), std::move(candidate.dependencies)};
    }

    void apply_merge(block_merge merge)
    {
        blocks.at(merge.target) = std::move(merge.block);
        blocks.at(merge.source).ops.clear();
        dependencies = std::move(merge.dependencies);
        for(const auto* op : blocks.at(merge.target).ops)
            assignment.update(op->info.ins, merge.target);
    }

    void coalesce_connected_blocks(const symbolic_op_plans& plans,
                                   std::set<std::pair<std::size_t, std::size_t>>& connections,
                                   std::size_t max_clones)
    {
        std::set<std::pair<std::size_t, std::size_t>> retries;
        while(not connections.empty())
        {
            auto connection = *connections.begin();
            connections.erase(connections.begin());
            auto [consumer_rank, input] = connection;
            const auto& consumer        = plans.values.at(consumer_rank).info.ins;
            const auto& producer        = consumer->inputs().at(input);
            auto consumer_block         = find_block(consumer);
            auto producer_block         = find_block(producer);
            assert(consumer_block.has_value() and producer_block.has_value());
            auto merged = find_merge(*consumer_block, *producer_block, plans, max_clones);
            if(merged.has_value())
            {
                auto target = merged->target;
                apply_merge(std::move(*merged));
                auto affected = blocks.at(target).find_connections(plans, assignment);
                auto next     = affected.upper_bound(connection);
                retries.insert(affected.begin(), next);
                connections.insert(next, affected.end());
            }
            if(connections.empty())
                connections.swap(retries);
        }
    }

    void coalesce_independent_blocks(const symbolic_op_plans& plans, std::size_t max_clones)
    {
        // Combine disconnected regions when doing so still forms a closed block. This avoids
        // separate select_module dispatches for independent branches that use the same compatible
        // root buckets.
        for(auto target : range(blocks.size()))
        {
            if(blocks.at(target).ops.empty())
                continue;
            for(auto source : range(target + 1, blocks.size()))
            {
                auto merged = find_merge(target, source, plans, max_clones);
                if(merged.has_value())
                {
                    auto merged_target = merged->target;
                    apply_merge(std::move(*merged));
                    auto connections = blocks.at(merged_target).find_connections(plans, assignment);
                    coalesce_connected_blocks(plans, connections, max_clones);
                    if(blocks.at(target).ops.empty())
                        break;
                }
            }
        }
    }

    bool can_propagate_padding_from_source_block(const symbolic_op_plan& plan,
                                                 const symbolic_op_plans& plans) const
    {
        const auto& inputs = plan.info.ins->inputs();
        auto input_indices = range(inputs.size());
        return any_of(input_indices, [&](auto input_index) {
            const auto& operand = plan.info.operands.at(input_index);
            if(not operand.pad_value.has_value() or operand.retained_slice_axes.empty())
                return false;
            auto source_block = find_block(inputs.at(input_index));
            if(not source_block.has_value() or *source_block >= blocks.size())
                return false;
            auto candidate = blocks.at(*source_block);
            candidate.ops.push_back(&plan);
            return candidate.resolved(plans, assignment).has_value();
        });
    }

    std::unordered_set<instruction_ref>
    find_absorbed_candidates(const symbolic_op_plans& plans) const
    {
        std::unordered_set<instruction_ref> result;
        std::unordered_set<instruction_ref> propagation_candidates;
        transform_if(
            plans.values.begin(),
            plans.values.end(),
            std::inserter(propagation_candidates, propagation_candidates.end()),
            [&](const auto& plan) {
                return not is_planned(plan.info.ins) and
                       plan.info.needs_deferred_specialization_block() and
                       can_propagate_padding_from_source_block(plan, plans);
            },
            [](const auto& plan) { return plan.info.ins; });
        std::for_each(blocks.begin(), blocks.end(), [&](const auto& block) {
            if(block.ops.empty())
                return;
            auto planned = block.instructions();
            std::for_each(plans.values.begin(), plans.values.end(), [&](const auto& plan) {
                if(is_planned(plan.info.ins) or not plan.info.needs_deferred_specialization_block())
                    return;
                if(contains(propagation_candidates, plan.info.ins))
                    return;
                std::unordered_set<instruction_ref> visited;
                auto reaches_specialization_block = fix<bool>([&](auto self,
                                                                  instruction_ref source) {
                    if(not visited.insert(source).second)
                        return false;
                    return any_of(source->outputs(), [&](auto output) {
                        if(not is_data_input(source, output, plans))
                            return false;
                        if(contains(planned, output))
                            return true;
                        return assignment.is_absorbable(output, planned, plans) and self(output);
                    });
                });
                if(reaches_specialization_block(plan.info.ins))
                    result.insert(plan.info.ins);
            });
        });
        return result;
    }

    void add_starting_blocks(const symbolic_op_plans& plans,
                             const std::vector<root_spec>& roots,
                             std::size_t max_clones)
    {
        std::for_each(plans.values.begin(), plans.values.end(), [&](const auto& plan) {
            if(not plan.info.starts_specialization_block())
                return;
            auto required_roots = find_instruction_roots(plan.info, roots, max_clones);
            if(required_roots.has_value())
                add({{&plan}, std::move(*required_roots), {}, {}});
        });
    }

    void coalesce_blocks(const symbolic_op_plans& plans, std::size_t max_clones)
    {
        update_dependencies(plans);
        if(not order().has_value())
            MIGRAPHX_THROW("SPLIT_SYM_DIM: cyclic block dependency");
        auto connections = find_all_connections(plans);
        coalesce_connected_blocks(plans, connections, max_clones);
        coalesce_independent_blocks(plans, max_clones);
    }

    void discover_deferred_blocks(const symbolic_op_plans& plans,
                                  const std::vector<root_spec>& roots,
                                  std::size_t max_clones)
    {
        auto absorbed_candidates = find_absorbed_candidates(plans);
        auto initial_block_count = blocks.size();
        std::for_each(plans.values.begin(), plans.values.end(), [&](const auto& plan) {
            if(is_planned(plan.info.ins) or not plan.info.needs_deferred_specialization_block() or
               contains(absorbed_candidates, plan.info.ins))
                return;
            auto required_roots = find_instruction_roots(plan.info, roots, max_clones);
            if(required_roots.has_value())
                add({{&plan}, std::move(*required_roots), {}, {}});
        });
        if(blocks.size() != initial_block_count)
            coalesce_blocks(plans, max_clones);
    }

    void finalize_order(const symbolic_op_plans& plans)
    {
        auto block_order = order();
        if(not block_order.has_value())
            MIGRAPHX_THROW("SPLIT_SYM_DIM: cyclic final block dependency");
        std::vector<block_plan> ordered_blocks;
        ordered_blocks.reserve(block_order->size());
        std::transform(block_order->begin(),
                       block_order->end(),
                       std::back_inserter(ordered_blocks),
                       [&](auto block_index) { return std::move(blocks.at(block_index)); });
        blocks = std::move(ordered_blocks);
        assignment.clear();
        for(auto block_index : range(blocks.size()))
            for(const auto* plan : blocks.at(block_index).ops)
                assignment.insert(plan->info.ins, block_index);
        update_dependencies(plans);
    }

    static block_dependency_graph discover(const symbolic_op_plans& plans,
                                           const std::vector<root_spec>& roots,
                                           std::size_t max_clones)
    {
        // Start with one block per operation that needs padding or a static rewrite. Attach the
        // root symbols needed to specialize that operation, then coalesce both producer-consumer
        // blocks and independent blocks when they can safely share one set of clones.
        block_dependency_graph result;
        result.add_starting_blocks(plans, roots, max_clones);
        result.coalesce_blocks(plans, max_clones);
        result.discover_deferred_blocks(plans, roots, max_clones);
        result.finalize_order(plans);
        return result;
    }

    void prepare_clone_infos(symbolic_op_plans& plans, const std::vector<root_spec>& roots) const
    {
        std::unordered_map<sym::expr, sym::expr> target_substitutions;
        std::transform(
            roots.begin(),
            roots.end(),
            std::inserter(target_substitutions, target_substitutions.end()),
            [](const auto& root) { return std::make_pair(root.root, root.target_symbol); });

        for(auto& plan : plans.values)
        {
            const auto& info     = plan.info;
            plan.dispatch_output = substitute_shape(info.get_output_shape(), target_substitutions);
            auto plan_block      = find_block(info.ins);
            if(not plan_block.has_value())
                continue;
            const auto& block = blocks.at(*plan_block);
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
                const auto& operand                         = info.operands.at(input_index);
                const resolved_input_padding_plan* resolved = nullptr;
                auto resolved_plan                          = block.resolved_inputs.find(info.ins);
                if(resolved_plan != block.resolved_inputs.end())
                    resolved = &resolved_plan->second.at(input_index);

                auto pad_value = operand.pad_value;
                auto masks     = operand.masks;
                std::optional<padding_variant_plan> padding_variant;
                if(resolved != nullptr)
                {
                    if(resolved->variant.has_value())
                        padding_variant = resolved->variant->plan;
                    if(resolved->fill.has_value())
                        pad_value = fill_value(*resolved->fill);
                }

                auto source = args.at(input_index);
                sliced_value input{{}, source, {}};
                const auto* source_plan          = plans.find(source);
                auto source_block                = find_block(source);
                const bool source_is_planned     = source_block.has_value();
                const bool source_in_same_block  = source_is_planned and source_block == plan_block;
                const bool source_in_other_block = source_is_planned and not source_in_same_block;

                // Start from the cross-block form; internal edges are simplified below.
                if(source_is_planned)
                {
                    assert(source_plan != nullptr);
                    input.slice_axes = source_plan->info.output_symbolic_axes;
                }

                // Also pad when fixed symbolic dims or strides change under the target
                // substitution, so the clone sees a static shape.
                bool emit_pad =
                    pad_value.has_value() or
                    needs_fixed_retarget(input_shapes.at(input_index), target_substitutions);
                if(source_in_same_block and pad_value.has_value())
                {
                    // Internal edges already carry bucket-sized values.
                    input.slice_axes.clear();
                    emit_pad = false;
                }
                else if(source_in_other_block and pad_value.has_value() and
                        not operand.retained_slice_axes.empty())
                {
                    // Cross-block edges retain axes where padding can affect the consumer and axes
                    // whose existing bucket padding must be discarded before applying a new fill.
                    std::vector<std::size_t> kept_axes;
                    std::copy_if(input.slice_axes.begin(),
                                 input.slice_axes.end(),
                                 std::back_inserter(kept_axes),
                                 [&](auto axis) {
                                     return contains(operand.retained_slice_axes, axis) or
                                            (resolved != nullptr and
                                             contains(resolved->repad_axes, axis));
                                 });
                    if(kept_axes.empty())
                    {
                        input.slice_axes.clear();
                        emit_pad = false;
                    }
                    else
                        input.slice_axes = std::move(kept_axes);
                }
                // A retarget-only pad has no reachable padded region, so its fill value is
                // irrelevant.
                if(emit_pad)
                    pad_value = pad_value.value_or(0.0f);
                else
                    pad_value.reset();
                plan.clone_inputs.push_back(
                    {std::move(input), pad_value, std::move(masks), std::move(padding_variant)});
            }
        }
    }
};

shape clone_parameter_shape(const shape& s,
                            const std::unordered_map<sym::expr, sym::expr>& subrange_symbols,
                            const std::unordered_map<sym::expr, sym::expr>& frozen_symbols)
{
    if(not s.symbolic())
        return s;
    auto result = substitute_shape(s, subrange_symbols);
    if(not result.symbolic())
        return result;
    auto dimensions = result.dyn_dims();
    std::transform(dimensions.begin(), dimensions.end(), dimensions.begin(), [&](const auto& d) {
        return shape::dynamic_dimension{d.sym_expr.subs(frozen_symbols)};
    });
    result = {result.type(), std::move(dimensions), result.dyn_strides()};
    if(s.is_fixed() and result.has_fixed_strides())
        return result.to_static();
    return result;
}

struct clone_output_case
{
    std::unordered_map<sym::expr, std::size_t> freeze;
    std::vector<shape> outputs;
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
        return all_of(range(expected.ndim()), [&](auto axis) {
            return expected.lens()[axis] <= 1 or expected.strides()[axis] == output.strides()[axis];
        });
    });
}

shape dispatch_shape_for_clones(const shape& planned,
                                const std::vector<clone_output_case>& clone_outputs,
                                std::size_t output_index)
{
    assert(not clone_outputs.empty());
    if(represents_clone_outputs(planned, clone_outputs, output_index))
        return planned;

    if(planned.symbolic())
    {
        for(const auto& clone_output : clone_outputs)
        {
            auto actual_layout =
                clone_output.outputs.at(output_index).with_lens(planned.type(), planned.dyn_dims());
            if(represents_clone_outputs(actual_layout, clone_outputs, output_index))
                return actual_layout;
        }
    }

    std::vector<shape> outputs;
    std::transform(clone_outputs.begin(),
                   clone_outputs.end(),
                   std::back_inserter(outputs),
                   [&](const auto& clone_output) { return clone_output.outputs.at(output_index); });
    MIGRAPHX_THROW("SPLIT_SYM_DIM: planned dispatch shape " + to_string(planned) +
                   " does not represent clone outputs " + to_string_range(outputs));
}

struct runtime_cache
{
    std::unordered_map<sym::expr, instruction_ref> extents;
    std::unordered_map<std::size_t, instruction_ref> indices;
    std::map<std::pair<shape::type_t, fill_kind>, instruction_ref> fills;

    instruction_ref get_or_add_extent(module& m,
                                      const sym::expr& expression,
                                      const std::vector<instruction_ref>& sources)
    {
        auto cached = extents.find(expression);
        if(cached != extents.end())
            return cached->second;
        auto result = m.add_instruction(
            make_op("eval_expr_from_shape",
                    {{"expressions", to_value(std::vector<sym::expr>{expression})}}),
            sources);
        return extents.emplace(expression, result).first->second;
    }

    instruction_ref get_or_add_index_literal(module& m, std::size_t n)
    {
        auto cached = indices.find(n);
        if(cached != indices.end())
            return cached->second;
        std::vector<int64_t> values(n);
        std::iota(values.begin(), values.end(), int64_t{0});
        auto result = m.add_literal(literal{shape{shape::int64_type, {n}}, values});
        return indices.emplace(n, result).first->second;
    }

    instruction_ref get_or_add_fill_literal(module& m, shape::type_t type, fill_kind fill)
    {
        auto key    = std::make_pair(type, fill);
        auto cached = fills.find(key);
        if(cached != fills.end())
            return cached->second;
        auto result =
            m.add_literal(literal{shape{type, {1}}, std::vector<float>{fill_value(fill)}});
        return fills.emplace(key, result).first->second;
    }

    instruction_ref add_mask(module& m,
                             instruction_ref input,
                             const axis_mask& mask,
                             const std::vector<instruction_ref>& sources,
                             const std::unordered_map<sym::expr, sym::expr>& fixed_substitutions)
    {
        const auto& s = input->get_shape();
        assert(not s.dynamic());
        assert(mask.axis < s.ndim());
        if(s.elements() == 0)
            return input;
        auto lens = s.lens();

        auto index =
            m.add_instruction(make_op("broadcast", {{"axis", mask.axis}, {"out_lens", lens}}),
                              get_or_add_index_literal(m, lens[mask.axis]));
        auto extent =
            m.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}),
                              get_or_add_extent(m, mask.extent.subs(fixed_substitutions), sources));
        auto valid = m.add_instruction(make_op("convert", {{"target_type", shape::bool_type}}),
                                       m.add_instruction(make_op("less"), index, extent));
        auto fill  = m.add_instruction(make_op("multibroadcast", {{"out_lens", lens}}),
                                       get_or_add_fill_literal(m, s.type(), mask.fill));
        return m.add_instruction(make_op("where"), valid, input, fill);
    }
};

struct block_input
{
    sliced_value clone_value;
    instruction_ref select_input;
};

struct replacement_context
{
    module& target;
    const block_dependency_graph& graph;
    const symbolic_op_plans& plans;
    std::unordered_map<instruction_ref, instruction_ref> values;

    void insert(instruction_ref source, instruction_ref replacement)
    {
        values.emplace(source, replacement);
    }

    instruction_ref at(instruction_ref source) const { return values.at(source); }

    void materialize(instruction_ref source)
    {
        if(contains(values, source))
            return;
        const auto* plan = plans.find(source);
        if(graph.is_planned(source))
        {
            assert(plan != nullptr);
            MIGRAPHX_THROW("SPLIT_SYM_DIM: block dependency " + source->name() + " at rank " +
                           std::to_string(plan->rank) + " was not specialized before use");
        }

        auto args = source->inputs();
        for(auto arg : args)
            materialize(arg);
        std::transform(
            args.begin(), args.end(), args.begin(), [&](instruction_ref arg) { return at(arg); });
        if(args == source->inputs())
        {
            insert(source, source);
            return;
        }

        auto result = target.add_instruction(source->get_operator(), args, source->module_inputs());
        if(not source->get_debug_symbols().empty())
            target.add_debug_symbols(result, source->get_debug_symbols());
        insert(source, result);
    }
};

struct cloned_values
{
    std::unordered_map<instruction_ref, instruction_ref> full;
    std::vector<std::pair<sliced_value, instruction_ref>> sliced;

    void set(instruction_ref source, instruction_ref value) { full[source] = value; }

    void set(const sliced_value& source, instruction_ref value)
    {
        if(source.slice_axes.empty())
            set(source.source, value);
        else
            sliced.emplace_back(source, value);
    }

    instruction_ref at(instruction_ref source) const { return full.at(source); }

    instruction_ref at(const sliced_value& source) const
    {
        if(source.slice_axes.empty())
            return at(source.source);
        auto found = std::find_if(
            sliced.begin(), sliced.end(), [&](const auto& value) { return value.first == source; });
        assert(found != sliced.end());
        return found->second;
    }
};

bool only_used_as_slice_metadata(instruction_ref ins)
{
    const auto& outputs = ins->outputs();
    return not outputs.empty() and std::all_of(outputs.begin(), outputs.end(), [&](auto output) {
        return contains({"slice", "dyn_slice"}, output->name()) and output->inputs().front() != ins;
    });
}

struct clone_context
{
    struct emitted_padding_variant
    {
        const padding_variant_plan* plan = nullptr;
        std::size_t step                 = 0;
        instruction_ref result;
    };

    module& clone_module;
    const block_dependency_graph& graph;
    const symbolic_op_plans& plans;
    cloned_values& values;
    const std::unordered_map<sym::expr, std::size_t>& freeze;
    const std::vector<instruction_ref>& runtime_extent_sources;
    const std::unordered_map<sym::expr, sym::expr>& fixed_substitutions;
    runtime_cache cache;
    std::unordered_map<instruction_ref, std::vector<instruction_ref>> reusable_pads;
    std::vector<emitted_padding_variant> padding_variants;

    instruction_ref get_or_add_pad(const operation& pad_op, instruction_ref input)
    {
        auto& candidates = reusable_pads[input];
        auto found =
            std::find_if(candidates.begin(), candidates.end(), [&](instruction_ref candidate) {
                return candidate->get_operator() == pad_op;
            });
        if(found != candidates.end())
            return *found;
        auto result = clone_module.add_instruction(pad_op, input);
        candidates.push_back(result);
        return result;
    }

    instruction_ref get_or_add_padding_variant(instruction_ref source,
                                               const padding_variant_plan& plan,
                                               std::size_t step)
    {
        if(step >= plan.steps.size() or plan.steps.at(step).source != source)
            MIGRAPHX_THROW("SPLIT_SYM_DIM: invalid concrete padding variant");
        auto found =
            std::find_if(padding_variants.begin(), padding_variants.end(), [&](const auto& entry) {
                return entry.plan->same_suffix(entry.step, plan, step);
            });
        if(found != padding_variants.end())
            return found->result;
        auto result = add_instruction(source, &plan, step);
        padding_variants.push_back({&plan, step, result});
        return result;
    }

    void apply_input_padding(instruction_ref source,
                             const symbolic_op_plan& clone_info,
                             const padding_variant_plan* padding_variant,
                             std::size_t variant_step,
                             std::vector<instruction_ref>& args)
    {
        const auto& clone_inputs = clone_info.clone_inputs;
        assert(clone_inputs.size() == args.size());
        std::vector<std::size_t> data_input_indices;
        auto input_indices = range(source->inputs().size());
        std::copy_if(input_indices.begin(),
                     input_indices.end(),
                     std::back_inserter(data_input_indices),
                     [&](auto index) { return not clone_info.is_shape_input(index); });
        assert(data_input_indices.size() == clone_inputs.size());

        const padding_variant_step* variant = nullptr;
        if(padding_variant != nullptr)
        {
            if(variant_step >= padding_variant->steps.size() or
               padding_variant->steps.at(variant_step).source != source)
                MIGRAPHX_THROW("SPLIT_SYM_DIM: invalid concrete padding variant");
            variant = &padding_variant->steps.at(variant_step);
        }

        bool applied_variant = variant == nullptr;
        for(auto&& [arg, clone_input, original_index] :
            views::zip(args, clone_inputs, data_input_indices))
        {
            std::optional<float> pad_value;
            if(variant != nullptr and variant->input == original_index)
            {
                applied_variant   = true;
                auto input_source = source->inputs().at(original_index);
                if(variant->boundary_pad_value.has_value())
                    pad_value = variant->boundary_pad_value;
                else
                {
                    auto next_step = variant_step + 1;
                    if(next_step >= padding_variant->steps.size())
                        MIGRAPHX_THROW("SPLIT_SYM_DIM: incomplete concrete padding variant");
                    arg = get_or_add_padding_variant(input_source, *padding_variant, next_step);
                }
            }
            else if(padding_variant == nullptr and clone_input.padding_variant.has_value())
            {
                arg = get_or_add_padding_variant(
                    source->inputs().at(original_index), *clone_input.padding_variant, 0);
            }
            else if(padding_variant == nullptr)
            {
                pad_value = clone_input.pad_value;
            }

            if(pad_value.has_value())
                arg = get_or_add_pad(make_op("fixed_pad", {{"value", *pad_value}}), arg);
        }
        if(not applied_variant)
            MIGRAPHX_THROW("SPLIT_SYM_DIM: concrete padding variant input is not data");

        for(auto&& [arg, clone_input] : views::zip(args, clone_inputs))
            arg = std::accumulate(
                clone_input.masks.begin(),
                clone_input.masks.end(),
                arg,
                [&](instruction_ref masked, const auto& mask) {
                    return cache.add_mask(
                        clone_module, masked, mask, runtime_extent_sources, fixed_substitutions);
                });
    }

    instruction_ref add_instruction(instruction_ref source,
                                    const padding_variant_plan* padding_variant,
                                    std::size_t variant_step)
    {
        const auto* plan          = plans.find(source);
        const bool source_planned = graph.is_planned(source);
        assert(not source_planned or plan != nullptr);
        const auto* clone_info = source_planned ? plan : nullptr;
        auto source_inputs     = plans.clone_inputs_for(source, source_planned);
        std::vector<instruction_ref> args;
        std::transform(source_inputs.begin(),
                       source_inputs.end(),
                       std::back_inserter(args),
                       [&](const sliced_value& input) { return values.at(input); });
        if(clone_info != nullptr)
            apply_input_padding(source, *clone_info, padding_variant, variant_step, args);

        op_freezer freezer;
        if(plan != nullptr)
            freezer = plan->info.freezer;

        instruction_ref clone;
        if(freezer)
            clone = freezer(
                clone_module, source, args, freezer_context{freeze, runtime_extent_sources});
        else
        {
            clone =
                clone_module.add_instruction(source->get_operator(), args, source->module_inputs());
        }

        if(clone->get_shape().dynamic())
            MIGRAPHX_THROW("SPLIT_SYM_DIM: clone body is not fully static: " + source->name());
        if(not source->get_debug_symbols().empty())
            clone_module.add_debug_symbols(clone, source->get_debug_symbols());
        return clone;
    }

    void emit(instruction_ref source) { values.set(source, add_instruction(source, nullptr, 0)); }
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

instruction_ref add_output_slice(module& m,
                                 const sliced_value& output,
                                 instruction_ref selected_output,
                                 const symbolic_op_info& info)
{
    if(info.output_symbolic_axes.empty())
        return selected_output;
    const auto& source_dims = output.source->get_shape().dyn_dims();
    std::vector<int64_t> axes;
    std::vector<sym::expr> end_expressions;
    for(auto axis : info.output_symbolic_axes)
    {
        assert(axis < source_dims.size());
        if(not contains(output.slice_axes, axis))
            continue;
        axes.push_back(static_cast<int64_t>(axis));
        end_expressions.push_back(source_dims.at(axis).sym_expr);
    }
    if(axes.empty())
        return selected_output;
    std::vector<sym::expr> start_expressions(axes.size(), sym::lit(int64_t{0}));
    auto sources = m.get_parameters();
    auto starts  = m.add_instruction(
        make_op("eval_expr_from_shape", {{"expressions", to_value(start_expressions)}}), sources);
    auto ends = m.add_instruction(
        make_op("eval_expr_from_shape", {{"expressions", to_value(end_expressions)}}), sources);
    auto result = m.add_instruction(make_op("dyn_slice",
                                            {{"axes", axes},
                                             {"starts", to_value(start_expressions)},
                                             {"ends", to_value(end_expressions)}}),
                                    selected_output,
                                    starts,
                                    ends);
    if(not output.source->get_debug_symbols().empty())
        m.add_debug_symbols(result, output.source->get_debug_symbols());
    return result;
}

struct block_frame
{
    std::vector<instruction_ref> body;
    std::vector<sliced_value> outputs;
    std::vector<block_input> inputs;
    std::map<std::string, std::size_t> params;
    std::vector<std::size_t> literals;
    std::vector<std::size_t> extent_sources;

    void add_input(sliced_value clone_value)
    {
        auto found = std::find_if(inputs.begin(), inputs.end(), [&](const auto& input) {
            return input.clone_value == clone_value;
        });
        if(found != inputs.end())
            return;
        inputs.push_back({std::move(clone_value), {}});
    }

    std::size_t input_index(const sliced_value& clone_value) const
    {
        auto found = std::find_if(inputs.begin(), inputs.end(), [&](const auto& input) {
            return input.clone_value == clone_value;
        });
        assert(found != inputs.end());
        return static_cast<std::size_t>(std::distance(inputs.begin(), found));
    }

    void add_required_output(const symbolic_op_plans& plans, sliced_value output)
    {
        if(output.slice_axes.empty())
            output = plans.full_output_for(output.source);
        if(not contains(outputs, output))
            outputs.push_back(std::move(output));
    }

    void collect_outputs(const module& m,
                         const block_plan& block,
                         const block_dependency_graph& graph,
                         const symbolic_op_plans& plans,
                         const std::unordered_set<instruction_ref>& planned_instructions,
                         const std::unordered_set<instruction_ref>& body_instructions)
    {
        for(auto output : m.get_returns())
            if(contains(body_instructions, output) and plans.contains(output))
                add_required_output(plans, {{}, output, {}});

        std::vector<instruction_ref> external_consumers;
        for(const auto* plan : block.ops)
            for(auto consumer : plan->info.ins->outputs())
                if(not contains(body_instructions, consumer) and
                   not contains(external_consumers, consumer))
                    external_consumers.push_back(consumer);
        for(auto consumer : external_consumers)
            for(auto input : plans.clone_inputs_for(consumer, graph.is_planned(consumer)))
                if(contains(planned_instructions, input.source))
                    add_required_output(plans, std::move(input));
    }

    void add_root_inputs(const block_plan& block,
                         const std::unordered_map<sym::expr, instruction_ref>& root_sources)
    {
        for(const auto* root : block.roots)
        {
            if(not contains(root_sources, root->root))
                MIGRAPHX_THROW("SPLIT_SYM_DIM: no parameter resolves block root " + root->name);
            sliced_value input{{}, root_sources.at(root->root), {}};
            add_input(input);
            auto index = input_index(input);
            if(not contains(extent_sources, index))
                extent_sources.push_back(index);
        }
    }

    void
    add_freezer_extent_inputs(const symbolic_op_plans& plans,
                              const std::unordered_map<sym::expr, instruction_ref>& root_sources)
    {
        std::vector<sym::expr> required_roots;
        for(auto ins : body)
        {
            const auto* plan = plans.find(ins);
            if(plan == nullptr)
                continue;
            for(const auto& expression : plan->info.runtime_extent_expressions)
            {
                auto variables = sym::find_variables(expression);
                std::copy_if(
                    variables.begin(),
                    variables.end(),
                    std::back_inserter(required_roots),
                    [&](const auto& variable) { return not contains(required_roots, variable); });
            }
        }
        std::sort(required_roots.begin(), required_roots.end(), [](const auto& x, const auto& y) {
            return x.to_string() < y.to_string();
        });
        for(const auto& root : required_roots)
        {
            if(not contains(root_sources, root))
                MIGRAPHX_THROW("SPLIT_SYM_DIM: no parameter resolves freezer extent root " +
                               root.to_string());
            sliced_value input{{}, root_sources.at(root), {}};
            add_input(input);
            auto index = input_index(input);
            if(not contains(extent_sources, index))
                extent_sources.push_back(index);
        }
    }

    void name_inputs(const module& m, std::size_t block_number)
    {
        auto parameter_names = m.get_parameter_names();
        std::unordered_set<std::string> used_names{parameter_names.begin(), parameter_names.end()};
        const std::string input_prefix = "#split_sym_dim_input_";
        std::size_t generated_suffix   = 0;
        for(auto input_index : range(inputs.size()))
        {
            auto source = inputs.at(input_index).clone_value.source;
            if(source->name() == "@param")
            {
                auto parameter_name =
                    source->get_operator().to_value().at("parameter").to<std::string>();
                params.emplace(std::move(parameter_name), input_index);
                continue;
            }
            if(source->name() == "@literal")
            {
                literals.push_back(input_index);
                ++generated_suffix;
                continue;
            }
            std::string name;
            do
            {
                name = input_prefix + std::to_string(block_number) + "_" +
                       std::to_string(generated_suffix++);
            } while(not used_names.insert(name).second);
            params.emplace(std::move(name), input_index);
        }
        if(params.size() + literals.size() != inputs.size())
            MIGRAPHX_THROW("SPLIT_SYM_DIM: failed to collect every block input");
    }

    void collect_body(instruction_ref ins,
                      const block_dependency_graph& graph,
                      const symbolic_op_plans& plans,
                      const std::unordered_set<instruction_ref>& planned_instructions,
                      std::unordered_set<instruction_ref>& body_instructions)
    {
        if(not body_instructions.insert(ins).second)
            return;
        auto ins_inputs = plans.clone_inputs_for(ins, graph.is_planned(ins));
        for(const auto& input : ins_inputs)
            if(graph.assignment.is_absorbable(input.source, planned_instructions, plans))
                collect_body(input.source, graph, plans, planned_instructions, body_instructions);
        body.push_back(ins);
        for(auto input : ins_inputs)
            add_input(std::move(input));
    }

    void resolve_inputs(replacement_context& replacements,
                        const std::vector<std::pair<sliced_value, instruction_ref>>& output_values)
    {
        for(auto& input : inputs)
        {
            if(input.clone_value.slice_axes.empty())
            {
                replacements.materialize(input.clone_value.source);
                input.select_input = replacements.at(input.clone_value.source);
                continue;
            }
            auto found =
                std::find_if(output_values.begin(), output_values.end(), [&](const auto& value) {
                    return value.first == input.clone_value;
                });
            if(found == output_values.end())
                MIGRAPHX_THROW("SPLIT_SYM_DIM: block output was not specialized before use");
            input.select_input = found->second;
        }
    }

    static auto build(const module& m,
                      const block_plan& block,
                      const block_dependency_graph& graph,
                      const symbolic_op_plans& plans,
                      std::size_t block_number,
                      const std::unordered_map<sym::expr, instruction_ref>& root_sources)
    {
        auto planned_instructions = block.instructions();

        block_frame result;
        std::unordered_set<instruction_ref> body_instructions;
        for(const auto* plan : block.ops)
            result.collect_body(
                plan->info.ins, graph, plans, planned_instructions, body_instructions);

        if(body_instructions.empty())
            return std::optional<block_frame>{};

        result.inputs.erase(std::remove_if(result.inputs.begin(),
                                           result.inputs.end(),
                                           [&](const auto& input) {
                                               return contains(body_instructions,
                                                               input.clone_value.source) and
                                                      input.clone_value.slice_axes.empty();
                                           }),
                            result.inputs.end());

        result.collect_outputs(m, block, graph, plans, planned_instructions, body_instructions);
        if(result.outputs.empty())
            return std::optional<block_frame>{};

        result.add_root_inputs(block, root_sources);
        result.add_freezer_extent_inputs(plans, root_sources);
        result.name_inputs(m, block_number);
        return std::optional<block_frame>{std::move(result)};
    }

    // Materialize one block specialization. Parameters retain the runtime subrange accepted by
    // this clone, while every operation in the body is emitted for the bucket's fixed target
    // extents. Inputs are padded or masked as planned, and operators such as dyn_slice and symbolic
    // broadcast are rewritten to static forms. No dynamic operation may remain in the emitted
    // clone body.
    clone_build
    build_clone(const std::string& name,
                const block_dependency_graph& graph,
                const symbolic_op_plans& plans,
                const std::unordered_map<sym::expr, std::size_t>& freeze,
                const std::unordered_map<sym::expr, shape::dynamic_dimension::interval>& subranges,
                const std::unordered_map<sym::expr, sym::expr>& fixed_substitutions) const
    {
        module clone_module{name};
        std::unordered_map<sym::expr, sym::expr> subrange_symbols;
        std::transform(subranges.begin(),
                       subranges.end(),
                       std::inserter(subrange_symbols, subrange_symbols.end()),
                       [](const auto& item) {
                           const auto& [root, interval] = item;
                           return std::make_pair(
                               root, sym::var(root.to_string(), {interval.min, interval.max}));
                       });
        std::unordered_map<sym::expr, sym::expr> frozen_symbols;
        transform_if(
            freeze.begin(),
            freeze.end(),
            std::inserter(frozen_symbols, frozen_symbols.end()),
            [&](const auto& item) { return not contains(subranges, item.first); },
            [](const auto& item) { return std::make_pair(item.first, sym::lit(item.second)); });
        cloned_values values;
        for(const auto& [parameter_name, input_index] : params)
        {
            const auto& input = inputs.at(input_index);
            auto parameter    = clone_module.add_parameter(
                parameter_name,
                clone_parameter_shape(
                    input.select_input->get_shape(), subrange_symbols, frozen_symbols));
            values.set(input.clone_value, parameter);
        }
        for(auto input_index : literals)
        {
            auto source = inputs.at(input_index).clone_value.source;
            values.set(source, clone_module.add_literal(source->get_literal()));
        }

        std::unordered_map<sym::expr, std::size_t> fixed_runtime_values;
        for(const auto& [root, interval] : subranges)
            if(interval.min == interval.max)
                fixed_runtime_values.emplace(root, interval.min);

        std::vector<instruction_ref> runtime_extent_inputs;
        std::transform(
            extent_sources.begin(),
            extent_sources.end(),
            std::back_inserter(runtime_extent_inputs),
            [&](std::size_t input_index) { return values.at(inputs.at(input_index).clone_value); });
        clone_context context{.clone_module           = clone_module,
                              .graph                  = graph,
                              .plans                  = plans,
                              .values                 = values,
                              .freeze                 = freeze,
                              .runtime_extent_sources = runtime_extent_inputs,
                              .fixed_substitutions    = fixed_substitutions};
        for(auto source : body)
            context.emit(source);
        match::find_matches(
            clone_module,
            fold_fixed_clone_evaluations{.fixed_runtime_values = fixed_runtime_values});

        std::vector<instruction_ref> clone_outputs;
        std::transform(outputs.begin(),
                       outputs.end(),
                       std::back_inserter(clone_outputs),
                       [&](const sliced_value& output) { return values.at(output.source); });
        auto dynamic_output =
            std::find_if(clone_outputs.begin(), clone_outputs.end(), [](instruction_ref output) {
                return output->get_shape().dynamic();
            });
        if(dynamic_output != clone_outputs.end())
            MIGRAPHX_THROW("SPLIT_SYM_DIM: clone output " + (*dynamic_output)->name() +
                           " is not fully static: " + to_string((*dynamic_output)->get_shape()));
        std::vector<shape> output_shapes;
        std::transform(clone_outputs.begin(),
                       clone_outputs.end(),
                       std::back_inserter(output_shapes),
                       [](instruction_ref output) { return output->get_shape(); });
        clone_module.add_return(clone_outputs);
        run_passes(clone_module, {dead_code_elimination{}});
        // Keep fixed parameters symbolic when an eval_expr_from_shape still reads them or
        // converting them would change an output layout.
        if(none_of(clone_module, [](const auto& ins) {
               return ins.name() == "eval_expr_from_shape" and not ins.outputs().empty();
           }))
        {
            auto static_clone = clone_module;
            for(auto parameter : static_clone.get_parameters())
            {
                const auto& s = parameter->get_shape();
                if(s.symbolic() and s.is_fixed() and s.has_fixed_strides())
                    instruction::replace(
                        parameter, parameter->get_operator(), s.to_static(), parameter->inputs());
            }
            if(static_clone.get_output_shapes() == output_shapes)
                clone_module = std::move(static_clone);
        }
        return {std::move(clone_module), {freeze, std::move(output_shapes)}};
    }

    void
    wire_select_module(module_pass_manager& mpm,
                       const symbolic_op_plans& plans,
                       std::vector<module> clones,
                       const std::vector<clone_output_case>& clone_outputs,
                       replacement_context& replacements,
                       std::vector<std::pair<sliced_value, instruction_ref>>& output_values) const
    {
        module& m = mpm.get_module();
        std::vector<instruction_ref> selection_inputs;
        std::transform(params.begin(),
                       params.end(),
                       std::back_inserter(selection_inputs),
                       [&](const auto& input) { return inputs.at(input.second).select_input; });
        std::vector<shape> body_output_shapes;
        auto output_indices = range(outputs.size());
        std::transform(output_indices.begin(),
                       output_indices.end(),
                       std::back_inserter(body_output_shapes),
                       [&](auto output_index) {
                           auto source = outputs.at(output_index).source;
                           return dispatch_shape_for_clones(
                               plans.at(source).dispatch_output, clone_outputs, output_index);
                       });
        auto selected_outputs =
            insert_select_module(mpm, m, selection_inputs, std::move(clones), body_output_shapes);

        for(std::size_t output_index = 0; output_index < outputs.size(); ++output_index)
        {
            const auto& output = outputs.at(output_index);
            auto sliced        = add_output_slice(
                m, output, selected_outputs.at(output_index), plans.at(output.source).info);
            output_values.emplace_back(output, sliced);
            if(output == plans.full_output_for(output.source))
                replacements.insert(output.source, sliced);
        }
    }
};

void specialize_blocks(module_pass_manager& mpm,
                       const block_dependency_graph& graph,
                       const symbolic_op_plans& plans,
                       const std::unordered_map<sym::expr, instruction_ref>& root_sources)
{
    // Replace each discovered block with one clone per cartesian product of its root buckets and a
    // select_module that dispatches to the compatible clone. The selected fixed-size outputs are
    // sliced back to their runtime extents before uses outside the block are rewired.
    module& m = mpm.get_module();
    replacement_context replacements{.target = m, .graph = graph, .plans = plans};
    std::vector<std::pair<sliced_value, instruction_ref>> output_values;
    auto original_outputs = m.get_returns();

    const auto& blocks = graph.blocks;
    auto block_numbers = range(blocks.size());
    for(auto&& [block_number, block] : views::zip(block_numbers, blocks))
    {
        auto frame = block_frame::build(m, block, graph, plans, block_number, root_sources);
        if(not frame.has_value())
            continue;
        frame->resolve_inputs(replacements, output_values);

        std::unordered_map<sym::expr, sym::expr> fixed_substitutions;
        transform_if(
            block.roots.begin(),
            block.roots.end(),
            std::inserter(fixed_substitutions, fixed_substitutions.end()),
            [](const auto* root) { return root->interval.min == root->interval.max; },
            [](const auto* root) { return std::make_pair(root->root, root->target_symbol); });
        std::vector<std::size_t> specialization_lens;
        std::transform(block.roots.rbegin(),
                       block.roots.rend(),
                       std::back_inserter(specialization_lens),
                       [](const auto* root) { return root->specializations.size(); });
        shape specialization_shape{shape::int64_type, specialization_lens};
        auto clone_count = specialization_shape.elements();
        std::vector<module> clones;
        std::vector<clone_output_case> clone_outputs;
        clones.reserve(clone_count);
        clone_outputs.reserve(clone_count);
        for(std::size_t clone_index = 0; clone_index < clone_count; ++clone_index)
        {
            auto specialization_indices = specialization_shape.multi(clone_index);
            std::reverse(specialization_indices.begin(), specialization_indices.end());
            std::unordered_map<sym::expr, std::size_t> freeze;
            std::unordered_map<sym::expr, shape::dynamic_dimension::interval> subranges;
            for(auto&& [root, specialization_index] :
                views::zip(block.roots, specialization_indices))
            {
                assert(specialization_index < root->specializations.size());
                auto specialization =
                    std::next(root->specializations.begin(), specialization_index);
                const auto& [target, runtime_range] = *specialization;
                freeze[root->root]                  = target;
                freeze[root->target_symbol]         = target;
                subranges[root->root]               = runtime_range;
            }
            auto name = m.name() + ":split_sym_dim_" + std::to_string(block_number) + "_" +
                        std::to_string(clone_index);
            auto built =
                frame->build_clone(name, graph, plans, freeze, subranges, fixed_substitutions);
            clones.push_back(std::move(built.clone));
            clone_outputs.push_back(std::move(built.output_case));
        }
        frame->wire_select_module(
            mpm, plans, std::move(clones), clone_outputs, replacements, output_values);
    }
    std::vector<instruction_ref> outputs;
    for(auto output : original_outputs)
        replacements.materialize(output);
    std::transform(original_outputs.begin(),
                   original_outputs.end(),
                   std::back_inserter(outputs),
                   [&](instruction_ref output) { return replacements.at(output); });
    m.replace_return(outputs);
    m.sort();
}

} // namespace

void split_sym_dim::apply(module_pass_manager& mpm) const
{
    module& m = mpm.get_module();

    if(not has_symbolic_param(m))
        return;

    // Collect independent symbols from parameter dimensions and partition each symbol's interval
    // into the target extents used to compile specialization clones.
    auto roots = collect_roots(m);
    if(not roots.has_value())
        return;

    // Rewrite shape expressions to read their runtime values directly from module parameters.
    auto root_sources = find_root_sources(m);
    match::find_matches(m,
                        resolve_symbolic_dimensions_of_match{.root_sources = root_sources,
                                                             .sources      = m.get_parameters()});
    match::find_matches(m,
                        find_eval_expr_from_intermediate_ins{.root_sources = root_sources,
                                                             .sources      = m.get_parameters()});
    run_passes(m, {eliminate_common_subexpression{}, dead_code_elimination{}});

    // Determine how each symbolic operation must be padded, masked, or rewritten for a fixed
    // target extent.
    auto symbolic_instructions = find_all(iterator_for(m), [](instruction_ref ins) {
        return ins->get_shape().symbolic() or any_of(ins->inputs(), [](instruction_ref input) {
                   return input->get_shape().symbolic();
               });
    });
    if(symbolic_instructions.empty())
        return;

    symbolic_op_plans plans;
    transform_if(
        symbolic_instructions.begin(),
        symbolic_instructions.end(),
        std::back_inserter(plans.values),
        [](instruction_ref ins) { return not starts_with(ins->name(), "@"); },
        [](instruction_ref ins) { return symbolic_op_plan{.info = analyze_instruction(ins)}; });
    plans.build_index();
    plans.remove_redundant_masks();

    // Seed blocks with operations that require specialization, then coalesce compatible connected
    // and independent blocks while preserving closed dependencies and the clone limit.
    auto graph = block_dependency_graph::discover(plans, *roots, max_clones);
    if(graph.blocks.empty())
        return;

    // Determine each block's boundary slices, padding, masks, and target-substituted output shapes.
    graph.prepare_clone_infos(plans, *roots);

    // Materialize one clone for each target combination, dispatch through select_module, slice
    // fixed-size outputs back to their runtime extents, and rewire uses outside each block.
    specialize_blocks(mpm, graph, plans, root_sources);
    run_passes(m, {dead_code_elimination{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
