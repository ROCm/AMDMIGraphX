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

struct clone_input_plan
{
    sliced_value clone_value;
    operand_plan operand;
};

struct symbolic_op_plan
{
    symbolic_op_info info;
    std::size_t rank = 0;
    std::optional<std::size_t> block;
    std::vector<clone_input_plan> clone_inputs;
    shape dispatch_output;
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
            masks.erase(std::remove_if(masks.begin(),
                                       masks.end(),
                                       [&](const auto& mask) {
                                           return source->second->info.zeros_contracted_region(
                                               mask);
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

bool has_bounded_interval(const sym::expr& expression)
{
    const auto interval = expression.eval_interval_default();
    return sym::to<int64_t>(interval.min) != std::numeric_limits<int64_t>::lowest() and
           sym::to<int64_t>(interval.max) != std::numeric_limits<int64_t>::max();
}

bool collect_root_dimension(const shape::dynamic_dimension& dimension,
                            std::unordered_map<sym::expr, root_spec>& root_specs)
{
    if(not dimension.is_symbolic())
        return true;
    if(dimension.sym_expr.name() != "variable")
        return not is_variable_axis(dimension);
    if(not has_bounded_interval(dimension.sym_expr))
        return false;

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
std::optional<std::vector<root_spec>> collect_roots(const module& m)
{
    std::unordered_map<sym::expr, root_spec> root_specs;
    for(auto parameter : m.get_parameters())
    {
        const auto& s = parameter->get_shape();
        if(s.dynamic() and not s.symbolic())
            return std::nullopt;
        if(not s.symbolic())
            continue;
        if(not all_of(s.dyn_dims(), [&](const auto& dimension) {
               return collect_root_dimension(dimension, root_specs);
           }))
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

const std::set<std::size_t>& find_instruction_block_dependencies(
    instruction_ref source,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    std::unordered_map<instruction_ref, std::set<std::size_t>>& cache)
{
    auto cached = cache.find(source);
    if(cached != cache.end())
        return cached->second;

    std::set<std::size_t> dependencies;
    auto source_info = info_for_instruction.find(source);
    if(source_info != info_for_instruction.end() and source_info->second->block.has_value())
    {
        dependencies.insert(*source_info->second->block);
    }
    else
    {
        const auto& inputs = source->inputs();
        auto input_indices = range(inputs.size());
        std::for_each(input_indices.begin(), input_indices.end(), [&](auto input_index) {
            if(source_info != info_for_instruction.end() and source_info->second->info.freezer and
               contains(source_info->second->info.shape_input_indices, input_index))
                return;
            const auto& found = find_instruction_block_dependencies(
                inputs.at(input_index), info_for_instruction, cache);
            dependencies.insert(found.begin(), found.end());
        });
    }
    return cache.emplace(source, std::move(dependencies)).first->second;
}

struct block_plan
{
    std::vector<symbolic_op_plan*> ops;
    std::vector<const root_spec*> roots;

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
};

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
        for(const auto& mask : operand.masks)
        {
            auto variables = sym::find_variables(mask.extent);
            required.insert(variables.begin(), variables.end());
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

bool absorbable_dependency(
    instruction_ref ins,
    const std::unordered_set<instruction_ref>& planned,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    if(contains(planned, ins))
        return true;
    auto found = info_for_instruction.find(ins);
    if(found != info_for_instruction.end() and found->second->block.has_value())
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
    if(found == info_for_instruction.end())
        return false;
    const auto& info = found->second->info;
    if(not info.supported or (not info.freezer and not s.is_fixed()))
        return false;
    const auto& inputs = ins->inputs();
    assert(inputs.size() == info.operands.size());
    return not any_of(views::zip(info.operands, inputs),
                      unpack([&](const auto& operand, auto input) {
                          return operand.pad_value.has_value() and
                                 absorbable_dependency(input, planned, info_for_instruction);
                      }));
}

bool block_is_closed(
    const block_plan& block,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    auto included = block.instructions();

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
        if(not absorbable_dependency(current, included, info_for_instruction))
            return not boundary_reaches_block(current);
        return all_of(current->inputs(), self);
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
                if(operand.pad_value.has_value() and not operand.retained_slice_axes.empty())
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
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    std::size_t max_clones)
{
    auto merged_roots = merge_block_roots(target, source, max_clones);
    if(not merged_roots.has_value())
        return false;
    block_plan result;
    result.ops = target.ops;
    result.ops.insert(result.ops.end(), source.ops.begin(), source.ops.end());
    if(not block_is_closed(result, info_for_instruction))
        return false;
    result.roots = std::move(*merged_roots);
    target       = std::move(result);
    return true;
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
    std::for_each(infos.begin(), infos.end(), [&](const auto& plan) {
        const auto& info   = plan.info;
        auto input_indices = range(info.ins->inputs().size());
        std::for_each(input_indices.begin(), input_indices.end(), [&](auto input) {
            insert_block_connection(result, info.ins, input, info_for_instruction);
        });
    });
    return result;
}

std::set<std::pair<std::size_t, std::size_t>> find_block_connections(
    const block_plan& block,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    std::set<std::pair<std::size_t, std::size_t>> result;
    std::for_each(block.ops.begin(), block.ops.end(), [&](const auto* plan) {
        const auto& info   = plan->info;
        auto input_indices = range(info.ins->inputs().size());
        std::for_each(input_indices.begin(), input_indices.end(), [&](auto input) {
            insert_block_connection(result, info.ins, input, info_for_instruction);
        });
        std::for_each(info.ins->outputs().begin(), info.ins->outputs().end(), [&](auto output) {
            const auto& inputs = output->inputs();
            auto output_inputs = range(inputs.size());
            std::for_each(output_inputs.begin(), output_inputs.end(), [&](auto input) {
                if(inputs.at(input) == info.ins)
                    insert_block_connection(result, output, input, info_for_instruction);
            });
        });
    });
    return result;
}

bool is_data_input(
    instruction_ref source,
    instruction_ref consumer,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    auto found   = info_for_instruction.find(consumer);
    auto indices = range(consumer->inputs().size());
    return any_of(indices, [&](auto index) {
        return consumer->inputs().at(index) == source and
               (found == info_for_instruction.end() or
                not contains(found->second->info.shape_input_indices, index));
    });
}

bool reaches_specialization_block(
    instruction_ref source,
    const std::unordered_set<instruction_ref>& planned,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    std::unordered_set<instruction_ref>& visited)
{
    if(not visited.insert(source).second)
        return false;
    return any_of(source->outputs(), [&](auto output) {
        if(not is_data_input(source, output, info_for_instruction))
            return false;
        if(contains(planned, output))
            return true;
        return absorbable_dependency(output, planned, info_for_instruction) and
               reaches_specialization_block(output, planned, info_for_instruction, visited);
    });
}

struct block_dependency_graph
{
    std::vector<block_plan> blocks;
    std::vector<std::set<std::size_t>> dependencies;

    std::size_t add(block_plan block)
    {
        auto result = blocks.size();
        blocks.push_back(std::move(block));
        dependencies.emplace_back();
        return result;
    }

    void update_dependencies(
        const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
    {
        std::unordered_map<instruction_ref, std::set<std::size_t>> cache;
        cache.reserve(info_for_instruction.size());
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
                            const auto& found = find_instruction_block_dependencies(
                                inputs.at(input_index), info_for_instruction, cache);
                            block_dependencies.insert(found.begin(), found.end());
                        });
                });
                assert(block.ops.front()->block.has_value());
                block_dependencies.erase(*block.ops.front()->block);
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
    std::optional<std::size_t> merge_blocks(
        std::size_t x,
        std::size_t y,
        const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
        std::size_t max_clones)
    {
        const auto target = std::min(x, y);
        const auto source = std::max(x, y);
        if(target == source or blocks.at(target).ops.empty() or blocks.at(source).ops.empty())
            return std::nullopt;

        auto candidate = *this;
        if(not merge_block_into(candidate.blocks.at(target),
                                candidate.blocks.at(source),
                                info_for_instruction,
                                max_clones))
            return std::nullopt;
        candidate.blocks.at(source).ops.clear();
        candidate.contract(target, source);
        if(not candidate.order().has_value())
            return std::nullopt;

        *this = std::move(candidate);
        for(auto* op : blocks.at(target).ops)
            op->block = target;
        return target;
    }

    void coalesce_connected_blocks(
        const std::vector<symbolic_op_plan>& infos,
        const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
        std::set<std::pair<std::size_t, std::size_t>>& connections,
        std::size_t max_clones)
    {
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
            assert(consumer_info->block.has_value() and producer_info->block.has_value());
            auto merged = merge_blocks(
                *consumer_info->block, *producer_info->block, info_for_instruction, max_clones);
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
        const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
        std::size_t max_clones)
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
                auto merged = merge_blocks(target, source, info_for_instruction, max_clones);
                if(merged.has_value())
                {
                    auto connections =
                        find_block_connections(blocks.at(*merged), info_for_instruction);
                    coalesce_connected_blocks(infos, info_for_instruction, connections, max_clones);
                    if(blocks.at(target).ops.empty())
                        break;
                }
            }
        }
    }

    std::unordered_set<instruction_ref>
    find_absorbed_candidates(const std::vector<symbolic_op_plan>& infos,
                             const std::unordered_map<instruction_ref, const symbolic_op_plan*>&
                                 info_for_instruction) const
    {
        std::unordered_set<instruction_ref> result;
        std::for_each(blocks.begin(), blocks.end(), [&](const auto& block) {
            if(block.ops.empty())
                return;
            auto planned = block.instructions();
            std::for_each(infos.begin(), infos.end(), [&](const auto& plan) {
                if(plan.block.has_value() or not plan.info.needs_deferred_specialization_block())
                    return;
                std::unordered_set<instruction_ref> visited;
                if(reaches_specialization_block(
                       plan.info.ins, planned, info_for_instruction, visited))
                    result.insert(plan.info.ins);
            });
        });
        return result;
    }

    static block_dependency_graph discover(
        std::vector<symbolic_op_plan>& infos,
        const std::vector<root_spec>& roots,
        const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
        std::size_t max_clones)
    {
        // Start with one block per operation that needs padding or a static rewrite. Attach the
        // root symbols needed to specialize that operation, then coalesce both producer-consumer
        // blocks and independent blocks when they can safely share one set of clones.
        block_dependency_graph result;
        std::for_each(infos.begin(), infos.end(), [&](auto& plan) {
            if(not plan.info.starts_specialization_block())
                return;
            auto required_roots = find_instruction_roots(plan.info, roots, max_clones);
            if(not required_roots.has_value())
                return;
            plan.block = result.add({{&plan}, std::move(*required_roots)});
        });

        result.update_dependencies(info_for_instruction);
        if(not result.order().has_value())
            MIGRAPHX_THROW("SPLIT_SYM_DIM: cyclic initial block dependency");

        auto connections = find_all_block_connections(infos, info_for_instruction);
        result.coalesce_connected_blocks(infos, info_for_instruction, connections, max_clones);
        result.coalesce_independent_blocks(infos, info_for_instruction, max_clones);

        auto absorbed_candidates = result.find_absorbed_candidates(infos, info_for_instruction);
        auto initial_block_count = result.blocks.size();
        std::for_each(infos.begin(), infos.end(), [&](auto& plan) {
            if(plan.block.has_value() or not plan.info.needs_deferred_specialization_block() or
               contains(absorbed_candidates, plan.info.ins))
                return;
            auto required_roots = find_instruction_roots(plan.info, roots, max_clones);
            if(not required_roots.has_value())
                return;
            plan.block = result.add({{&plan}, std::move(*required_roots)});
        });

        if(result.blocks.size() != initial_block_count)
        {
            result.update_dependencies(info_for_instruction);
            if(not result.order().has_value())
                MIGRAPHX_THROW("SPLIT_SYM_DIM: cyclic deferred block dependency");
            connections = find_all_block_connections(infos, info_for_instruction);
            result.coalesce_connected_blocks(infos, info_for_instruction, connections, max_clones);
            result.coalesce_independent_blocks(infos, info_for_instruction, max_clones);
        }

        auto block_order = result.order();
        if(not block_order.has_value())
            MIGRAPHX_THROW("SPLIT_SYM_DIM: cyclic final block dependency");
        std::vector<block_plan> ordered_blocks;
        ordered_blocks.reserve(block_order->size());
        std::transform(block_order->begin(),
                       block_order->end(),
                       std::back_inserter(ordered_blocks),
                       [&](auto block_index) { return std::move(result.blocks.at(block_index)); });
        result.blocks        = std::move(ordered_blocks);
        auto ordered_indices = range(result.blocks.size());
        std::for_each(ordered_indices.begin(), ordered_indices.end(), [&](auto block_index) {
            auto& ops = result.blocks.at(block_index).ops;
            std::for_each(ops.begin(), ops.end(), [&](auto* plan) { plan->block = block_index; });
        });
        result.update_dependencies(info_for_instruction);
        return result;
    }
};

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
    std::transform(roots.begin(),
                   roots.end(),
                   std::inserter(target_substitutions, target_substitutions.end()),
                   [](const auto& root) { return std::make_pair(root.root, root.target_symbol); });

    for(auto& plan : infos)
    {
        const auto& info     = plan.info;
        plan.dispatch_output = substitute_shape(info.get_output_shape(), target_substitutions);
        if(not plan.block.has_value())
            continue;
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
            sliced_value input{{}, source, {}};
            auto source_info             = info_for_instruction.find(source);
            const bool source_is_planned = source_info != info_for_instruction.end() and
                                           source_info->second->block.has_value();
            const bool source_in_same_block =
                source_is_planned and source_info->second->block == plan.block;
            const bool source_in_other_block = source_is_planned and not source_in_same_block;

            // Start from the cross-block form; internal edges are simplified below.
            if(source_is_planned)
                input.slice_axes = source_info->second->info.output_symbolic_axes;

            // Also pad when fixed symbolic dims or strides change under the target substitution,
            // so the clone sees a static shape.
            bool emit_pad =
                operand.pad_value.has_value() or
                needs_fixed_retarget(input_shapes.at(input_index), target_substitutions);
            if(source_in_same_block and operand.pad_value.has_value())
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
    }
}

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
            result.push_back({{}, input, {}});
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
    if(s.elements() == 0)
        return input;
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

sliced_value full_output_for(
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    instruction_ref source)
{
    sliced_value result{{}, source, {}};
    result.slice_axes = info_for_instruction.at(source)->info.output_symbolic_axes;
    return result;
}

std::size_t add_block_input(std::vector<block_input>& inputs, sliced_value clone_value)
{
    auto found = std::find_if(inputs.begin(), inputs.end(), [&](const auto& input) {
        return input.clone_value == clone_value;
    });
    if(found != inputs.end())
        return static_cast<std::size_t>(std::distance(inputs.begin(), found));
    inputs.push_back({std::move(clone_value), {}});
    return inputs.size() - 1;
}

instruction_ref resolve_replacement(
    module& m,
    instruction_ref source,
    std::unordered_map<instruction_ref, instruction_ref>& replacements,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction)
{
    auto found = replacements.find(source);
    if(found != replacements.end())
        return found->second;
    auto info = info_for_instruction.find(source);
    if(info != info_for_instruction.end() and info->second->block.has_value())
        MIGRAPHX_THROW("SPLIT_SYM_DIM: block dependency " + source->name() + " at rank " +
                       std::to_string(info->second->rank) + " was not specialized before use");

    auto args = source->inputs();
    std::transform(args.begin(), args.end(), args.begin(), [&](instruction_ref arg) {
        return resolve_replacement(m, arg, replacements, info_for_instruction);
    });
    if(args == source->inputs())
    {
        replacements.emplace(source, source);
        return source;
    }

    auto result = m.add_instruction(source->get_operator(), args, source->module_inputs());
    if(not source->get_debug_symbols().empty())
        m.add_debug_symbols(result, source->get_debug_symbols());
    replacements.emplace(source, result);
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

    void add_required_output(
        const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
        sliced_value output)
    {
        if(output.slice_axes.empty())
            output = full_output_for(info_for_instruction, output.source);
        if(not contains(outputs, output))
            outputs.push_back(std::move(output));
    }

    void collect_outputs(
        const module& m,
        const block_plan& block,
        const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
        const std::unordered_set<instruction_ref>& planned_instructions,
        const std::unordered_set<instruction_ref>& body_instructions)
    {
        for(auto output : m.get_returns())
            if(contains(body_instructions, output) and contains(info_for_instruction, output))
                add_required_output(info_for_instruction, {{}, output, {}});

        std::vector<instruction_ref> external_consumers;
        for(const auto* plan : block.ops)
            for(auto consumer : plan->info.ins->outputs())
                if(not contains(body_instructions, consumer) and
                   not contains(external_consumers, consumer))
                    external_consumers.push_back(consumer);
        for(auto consumer : external_consumers)
            for(auto input : clone_inputs_for(info_for_instruction, consumer))
                if(contains(planned_instructions, input.source))
                    add_required_output(info_for_instruction, std::move(input));
    }

    void add_root_inputs(const block_plan& block,
                         const std::unordered_map<sym::expr, instruction_ref>& root_sources)
    {
        for(const auto* root : block.roots)
        {
            if(not contains(root_sources, root->root))
                MIGRAPHX_THROW("SPLIT_SYM_DIM: no parameter resolves block root " + root->name);
            auto input_index = add_block_input(inputs, {{}, root_sources.at(root->root), {}});
            if(not contains(extent_sources, input_index))
                extent_sources.push_back(input_index);
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

    void collect_body(
        instruction_ref ins,
        const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
        const std::unordered_set<instruction_ref>& planned_instructions,
        std::unordered_set<instruction_ref>& body_instructions)
    {
        if(not body_instructions.insert(ins).second)
            return;
        auto ins_inputs = clone_inputs_for(info_for_instruction, ins);
        for(const auto& input : ins_inputs)
            if(absorbable_dependency(input.source, planned_instructions, info_for_instruction))
                collect_body(
                    input.source, info_for_instruction, planned_instructions, body_instructions);
        body.push_back(ins);
        for(auto input : ins_inputs)
            add_block_input(inputs, std::move(input));
    }

    void resolve_inputs(
        module& m,
        std::unordered_map<instruction_ref, instruction_ref>& replacements,
        const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
        const std::vector<std::pair<sliced_value, instruction_ref>>& output_values)
    {
        for(auto& input : inputs)
        {
            if(input.clone_value.slice_axes.empty())
            {
                input.select_input = resolve_replacement(
                    m, input.clone_value.source, replacements, info_for_instruction);
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
};

std::optional<block_frame> find_block_frame(
    const module& m,
    const block_plan& block,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    std::size_t block_number,
    const std::unordered_map<sym::expr, instruction_ref>& root_sources)
{
    auto planned_instructions = block.instructions();

    block_frame result;
    std::unordered_set<instruction_ref> body_instructions;
    for(const auto* plan : block.ops)
        result.collect_body(
            plan->info.ins, info_for_instruction, planned_instructions, body_instructions);

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

    result.collect_outputs(m, block, info_for_instruction, planned_instructions, body_instructions);
    if(result.outputs.empty())
        return std::nullopt;

    result.add_root_inputs(block, root_sources);
    result.name_inputs(m, block_number);
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
            assert(clone_info->clone_inputs.size() == args.size());
            for(auto&& [arg, input] : views::zip(args, clone_info->clone_inputs))
            {
                if(input.operand.pad_value.has_value())
                    arg = add_or_reuse_pad(
                        clone_module,
                        make_op("fixed_pad", {{"value", *input.operand.pad_value}}),
                        arg,
                        reusable_pads);
                for(const auto& mask : input.operand.masks)
                    arg = add_runtime_mask(clone_module,
                                           arg,
                                           mask,
                                           runtime_extent_sources,
                                           fixed_substitutions,
                                           cache);
            }
        }

        op_freezer freezer;
        if(found != info_for_instruction.end())
            freezer = found->second->info.freezer;

        instruction_ref clone;
        if(freezer)
            clone = freezer(clone_module, source, args, freeze);
        else
        {
            clone =
                clone_module.add_instruction(source->get_operator(), args, source->module_inputs());
        }

        if(clone->get_shape().dynamic())
            MIGRAPHX_THROW("SPLIT_SYM_DIM: clone body is not fully static: " + source->name());
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
    std::unordered_map<instruction_ref, instruction_ref> clone_map;
    std::vector<std::pair<sliced_value, instruction_ref>> input_values;
    for(const auto& [parameter_name, input_index] : frame.params)
    {
        const auto& input = frame.inputs.at(input_index);
        auto parameter    = clone_module.add_parameter(
            parameter_name,
            clone_parameter_shape(
                input.select_input->get_shape(), subrange_symbols, frozen_symbols));
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
    clone_context context{.clone_module           = clone_module,
                          .info_for_instruction   = info_for_instruction,
                          .clone_map              = clone_map,
                          .input_values           = input_values,
                          .freeze                 = freeze,
                          .runtime_extent_sources = runtime_extent_sources,
                          .fixed_substitutions    = fixed_substitutions};
    for(auto source : frame.body)
        context.emit(source);
    match::find_matches(clone_module,
                        fold_fixed_clone_evaluations{.fixed_runtime_values = fixed_runtime_values});

    std::vector<instruction_ref> clone_outputs;
    std::transform(frame.outputs.begin(),
                   frame.outputs.end(),
                   std::back_inserter(clone_outputs),
                   [&](const sliced_value& output) { return clone_map.at(output.source); });
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
    // Keep fixed parameters symbolic when an eval_expr_from_shape still reads them or converting
    // them would change an output layout.
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

void wire_select_module(
    module_pass_manager& mpm,
    const block_frame& frame,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    std::vector<module> clones,
    const std::vector<clone_output_case>& clone_outputs,
    std::unordered_map<instruction_ref, instruction_ref>& replacements,
    std::vector<std::pair<sliced_value, instruction_ref>>& output_values)
{
    module& m = mpm.get_module();
    std::vector<instruction_ref> selection_inputs;
    std::transform(frame.params.begin(),
                   frame.params.end(),
                   std::back_inserter(selection_inputs),
                   [&](const auto& input) { return frame.inputs.at(input.second).select_input; });
    std::vector<shape> body_output_shapes;
    auto output_indices = range(frame.outputs.size());
    std::transform(
        output_indices.begin(),
        output_indices.end(),
        std::back_inserter(body_output_shapes),
        [&](auto output_index) {
            auto source = frame.outputs.at(output_index).source;
            return dispatch_shape_for_clones(
                info_for_instruction.at(source)->dispatch_output, clone_outputs, output_index);
        });
    auto selected_outputs =
        insert_select_module(mpm, m, selection_inputs, std::move(clones), body_output_shapes);

    for(std::size_t output_index = 0; output_index < frame.outputs.size(); ++output_index)
    {
        const auto& output = frame.outputs.at(output_index);
        auto sliced        = add_output_slice(m,
                                              output,
                                              selected_outputs.at(output_index),
                                              info_for_instruction.at(output.source)->info);
        output_values.emplace_back(output, sliced);
        if(output == full_output_for(info_for_instruction, output.source))
            replacements.emplace(output.source, sliced);
    }
}

void specialize_blocks(
    module_pass_manager& mpm,
    const std::vector<block_plan>& blocks,
    const std::unordered_map<instruction_ref, const symbolic_op_plan*>& info_for_instruction,
    const std::unordered_map<sym::expr, instruction_ref>& root_sources)
{
    // Replace each discovered block with one clone per cartesian product of its root buckets and a
    // select_module that dispatches to the compatible clone. The selected fixed-size outputs are
    // sliced back to their runtime extents before uses outside the block are rewired.
    module& m = mpm.get_module();
    std::unordered_map<instruction_ref, instruction_ref> replacements;
    std::vector<std::pair<sliced_value, instruction_ref>> output_values;
    auto original_outputs = m.get_returns();

    auto block_numbers = range(blocks.size());
    for(auto&& [block_number, block] : views::zip(block_numbers, blocks))
    {
        auto frame = find_block_frame(m, block, info_for_instruction, block_number, root_sources);
        if(not frame.has_value())
            continue;
        frame->resolve_inputs(m, replacements, info_for_instruction, output_values);

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
            auto name  = m.name() + ":split_sym_dim_" + std::to_string(block_number) + "_" +
                         std::to_string(clone_index);
            auto built = build_clone(
                name, *frame, info_for_instruction, freeze, subranges, fixed_substitutions);
            clones.push_back(std::move(built.clone));
            clone_outputs.push_back(std::move(built.output_case));
        }
        wire_select_module(mpm,
                           *frame,
                           info_for_instruction,
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
                       return resolve_replacement(m, output, replacements, info_for_instruction);
                   });
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
        auto& plan = infos.at(rank);
        plan.rank  = rank;
        info_for_instruction.emplace(plan.info.ins, &plan);
    }
    remove_redundant_masks(infos, info_for_instruction);

    // Seed blocks with operations that require specialization, then coalesce compatible connected
    // and independent blocks while preserving closed dependencies and the clone limit.
    auto graph = block_dependency_graph::discover(infos, *roots, info_for_instruction, max_clones);
    if(graph.blocks.empty())
        return;

    // Determine each block's boundary slices, padding, masks, and target-substituted output shapes.
    prepare_clone_infos(infos, info_for_instruction, *roots);

    // Materialize one clone for each target combination, dispatch through select_module, slice
    // fixed-size outputs back to their runtime extents, and rewire uses outside each block.
    specialize_blocks(mpm, graph.blocks, info_for_instruction, root_sources);
    run_passes(m, {dead_code_elimination{}});
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
