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
#include <migraphx/op/select_module.hpp>
#include <migraphx/op/get_tuple_elem.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/builtin.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/instruction_traversal.hpp>
#include <migraphx/ranges.hpp>
#include <algorithm>
#include <limits>
#include <optional>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

namespace {

using module_metadata    = select_module::module_metadata;
using output_location    = select_module::output_location;
using parameter_metadata = select_module::parameter_metadata;

std::optional<std::size_t>
evaluate_dimension(const sym::expr& expression,
                   const std::unordered_map<sym::expr, std::size_t>& values)
{
    std::unordered_map<sym::expr, sym::expr> substitutions;
    std::transform(
        values.begin(),
        values.end(),
        std::inserter(substitutions, substitutions.end()),
        [](const auto& item) { return std::make_pair(item.first, sym::lit(item.second)); });
    auto value = sym::fixed_value(expression.subs(substitutions));
    if(not value.has_value())
        return std::nullopt;
    auto integer = sym::to<int64_t>(*value);
    if(integer < 0)
        MIGRAPHX_THROW("SELECT_MODULE: output dimension is negative");
    return static_cast<std::size_t>(integer);
}

// The ref target wraps every operator in ref::op, which reports the wrapped one in to_value.
std::optional<std::size_t> tuple_element_index(instruction_ref ins)
{
    if(ins->name() == "get_tuple_elem")
        return any_cast<get_tuple_elem>(ins->get_operator()).index;
    if(ins->name() != "ref::op")
        return std::nullopt;
    auto v = ins->get_operator().to_value();
    if(v.at("name").to<std::string>() != "get_tuple_elem")
        return std::nullopt;
    return v.at("operator").at("index").to<std::size_t>();
}

// The path of a tuple output parameter that a return writes, found by following the return's
// aliases back through every get_tuple_elem that reads the parameter.
std::optional<std::vector<std::size_t>> returned_tuple_path(instruction_ref ret,
                                                            instruction_ref parameter)
{
    using tuple_path = std::vector<std::size_t>;
    return fix<std::optional<tuple_path>>(
        [&](auto self, instruction_ref ins, tuple_path projections) {
            if(ins == parameter)
                return std::optional<tuple_path>{std::move(projections)};
            if(auto index = tuple_element_index(ins))
            {
                if(ins->inputs().empty())
                    return std::optional<tuple_path>{};
                projections.insert(projections.begin(), *index);
                return self(ins->inputs().front(), std::move(projections));
            }
            auto aliases = ins->get_operator().output_alias(to_shapes(ins->inputs()));
            if(ins->name() == "select_module" and not projections.empty())
            {
                auto output_index = projections.front();
                if(output_index >= aliases.size())
                    return std::optional<tuple_path>{};
                projections.erase(projections.begin());
                return self(ins->inputs().at(aliases.at(output_index)), std::move(projections));
            }
            for(auto index : aliases)
            {
                auto result = self(ins->inputs().at(index), projections);
                if(result.has_value())
                    return result;
            }
            return std::optional<tuple_path>{};
        })(ret, tuple_path{});
}

bool aliases_parameter(instruction_ref ret, instruction_ref parameter)
{
    return contains(resolve_output_aliases(ret), parameter);
}

std::vector<std::vector<std::size_t>> parameter_leaf_paths(const shape& parameter_shape)
{
    std::vector<std::vector<std::size_t>> result;
    std::vector<std::size_t> path;
    fix([&](auto self, const shape& current) {
        if(current.type() != shape::tuple_type)
        {
            result.push_back(path);
            return;
        }
        const auto& sub_shapes = current.sub_shapes();
        for(auto i : range(sub_shapes.size()))
        {
            path.push_back(i);
            self(sub_shapes[i]);
            path.pop_back();
        }
    })(parameter_shape);
    return result;
}

const shape* shape_at_path(const shape& root, const std::vector<std::size_t>& path)
{
    const shape* result = &root;
    for(auto index : path)
    {
        if(result->type() != shape::tuple_type or index >= result->sub_shapes().size())
            return nullptr;
        result = &result->sub_shapes()[index];
    }
    return result;
}

// The caller-buffer locations an output parameter writes. A return can be a parameter leaf, a
// whole tuple, or a subtuple, and tuple leaves need not be in return order.
std::vector<output_location> output_locations(const std::string& name,
                                              instruction_ref parameter,
                                              const std::vector<instruction_ref>& returns,
                                              const std::vector<shape>& output_shapes)
{
    const auto& parameter_shape = parameter->get_shape();
    constexpr auto missing      = std::numeric_limits<std::size_t>::max();
    auto leaf_paths             = parameter_leaf_paths(parameter_shape);
    std::vector<output_location> result(leaf_paths.size(), output_location{missing, {}});
    auto return_indices = range(returns.size());
    migraphx::for_each(returns.begin(),
                       returns.end(),
                       return_indices.begin(),
                       [&](instruction_ref ret, std::size_t r) {
                           if(not aliases_parameter(ret, parameter))
                               return;
                           auto path = returned_tuple_path(ret, parameter);
                           if(not path.has_value())
                               return;
                           for(auto leaf_index : range(leaf_paths.size()))
                           {
                               const auto& leaf_path = leaf_paths[leaf_index];
                               if(path->size() > leaf_path.size() or
                                  not std::equal(path->begin(), path->end(), leaf_path.begin()))
                                   continue;
                               std::vector<std::size_t> output_path(
                                   leaf_path.begin() + path->size(), leaf_path.end());
                               const auto* output_shape =
                                   shape_at_path(output_shapes.at(r), output_path);
                               const auto* leaf_shape = shape_at_path(parameter_shape, leaf_path);
                               if(output_shape == nullptr or leaf_shape == nullptr or
                                  output_shape->type() != leaf_shape->type())
                                   continue;
                               result[leaf_index] = output_location{r, std::move(output_path)};
                           }
                       });
    if(std::any_of(result.begin(), result.end(), [&](const auto& x) { return x.index == missing; }))
    {
        if(parameter_shape.type() == shape::tuple_type)
            MIGRAPHX_THROW("SELECT_MODULE: tuple output parameter \"" + name +
                           "\" is not aliased by module returns for every subobject");
        MIGRAPHX_THROW("SELECT_MODULE: output parameter \"" + name +
                       "\" does not alias a module output");
    }
    return result;
}

module_metadata
make_module_metadata(const select_module& select, module_ref mod, std::size_t module_index)
{
    module_metadata result;
    result.module_index = module_index;
    auto parameters     = mod->get_parameters();
    std::unordered_map<std::string, instruction_ref> parameter_by_name;
    std::transform(parameters.begin(),
                   parameters.end(),
                   std::inserter(parameter_by_name, parameter_by_name.end()),
                   [](instruction_ref parameter) {
                       return std::make_pair(
                           any_cast<builtin::param>(parameter->get_operator()).parameter,
                           parameter);
                   });

    auto input_names = select.get_input_parameter_names(mod);
    std::transform(input_names.begin(),
                   input_names.end(),
                   std::back_inserter(result.inputs),
                   [&](const std::string& name) {
                       auto parameter = parameter_by_name.at(name);
                       return parameter_metadata{name, parameter->get_shape(), {}};
                   });

    auto output_names = select.get_output_parameter_names(mod);
    auto returns      = mod->get_returns();
    if(not output_names.empty() and returns.size() != select.num_outputs())
        MIGRAPHX_THROW("SELECT_MODULE: output allocation count does not match module outputs");
    std::transform(
        output_names.begin(),
        output_names.end(),
        std::back_inserter(result.outputs),
        [&](const std::string& name) {
            auto parameter = parameter_by_name.at(name);
            return parameter_metadata{
                name,
                parameter->get_shape(),
                output_locations(name, parameter, returns, select.output_dyn_shapes.sub_shapes())};
        });
    return result;
}

// The input positions whose parameter differs between the candidates, which are the only ones
// that have to be compared to tell the candidates apart.
std::vector<std::size_t> selector_indices(const std::vector<module_metadata>& modules,
                                          const module_metadata& candidate)
{
    auto indices = range(candidate.inputs.size());
    std::vector<std::size_t> result;
    std::copy_if(
        indices.begin(), indices.end(), std::back_inserter(result), [&](std::size_t index) {
            const auto& expected = candidate.inputs[index];
            return std::any_of(modules.begin(), modules.end(), [&](const module_metadata& other) {
                if(index >= other.inputs.size())
                    return true;
                const auto& input = other.inputs[index];
                return input.name != expected.name or
                       input.parameter_shape != expected.parameter_shape;
            });
        });
    return result;
}

std::vector<std::size_t> shared_dynamic_indices(const module_metadata& candidate)
{
    auto indices = range(candidate.inputs.size());
    std::vector<std::size_t> result;
    std::copy_if(
        indices.begin(), indices.end(), std::back_inserter(result), [&](std::size_t index) {
            return candidate.inputs[index].parameter_shape.dynamic() and
                   not contains(candidate.selector_indices, index);
        });
    return result;
}

shape resolve_output_shape(const shape& output,
                           const std::unordered_map<sym::expr, std::size_t>& values,
                           const shape* physical = nullptr)
{
    if(output.type() == shape::tuple_type)
    {
        if(physical != nullptr and (physical->type() != shape::tuple_type or
                                    physical->sub_shapes().size() != output.sub_shapes().size()))
            MIGRAPHX_THROW("SELECT_MODULE: physical output tuple does not match declared output");
        std::vector<shape> sub_shapes;
        auto indices = range(output.sub_shapes().size());
        std::transform(indices.begin(), indices.end(), std::back_inserter(sub_shapes), [&](auto i) {
            const shape* physical_sub =
                physical == nullptr ? nullptr : &physical->sub_shapes().at(i);
            return resolve_output_shape(output.sub_shapes().at(i), values, physical_sub);
        });
        return shape{sub_shapes};
    }

    if(physical != nullptr and
       (physical->type() != output.type() or physical->ndim() != output.ndim()))
        MIGRAPHX_THROW("SELECT_MODULE: physical output does not match declared output");

    std::vector<std::size_t> lens;
    if(not output.dynamic())
    {
        lens = output.lens();
    }
    else
    {
        lens.reserve(output.ndim());
        auto axes = range(output.ndim());
        std::transform(axes.begin(), axes.end(), std::back_inserter(lens), [&](auto axis) {
            const auto& dim = output.dyn_dims().at(axis);
            if(dim.is_fixed())
                return dim.get_interval().min;
            if(dim.is_symbolic())
            {
                auto value = evaluate_dimension(dim.sym_expr, values);
                if(value.has_value())
                    return *value;
            }
            return physical == nullptr ? dim.get_interval().max : physical->lens().at(axis);
        });
    }

    if(physical == nullptr)
        return shape{output.type(), lens};
    if(any_of(range(lens.size()),
              [&](auto axis) { return lens.at(axis) > physical->lens().at(axis); }))
        MIGRAPHX_THROW("SELECT_MODULE: logical output exceeds physical output");
    return shape{output.type(), std::move(lens), physical->strides()};
}

argument reshape_output(const argument& arg, const shape& output_shape)
{
    if(output_shape.type() != shape::tuple_type)
        return arg.reshape(output_shape);
    auto sub_args = arg.get_sub_objects();
    if(sub_args.size() != output_shape.sub_shapes().size())
        MIGRAPHX_THROW("SELECT_MODULE: output tuple does not match declared output");
    std::vector<argument> result;
    std::transform(sub_args.begin(),
                   sub_args.end(),
                   output_shape.sub_shapes().begin(),
                   std::back_inserter(result),
                   &reshape_output);
    return argument{result};
}

} // namespace

std::unordered_map<sym::expr, std::size_t>
select_module::runtime_symbol_values(const std::vector<argument>& args) const
{
    assert(input_shapes.size() <= args.size());
    if(not std::equal(input_shapes.begin(),
                      input_shapes.end(),
                      args.begin(),
                      [](const shape& expected, const argument& arg) {
                          return compatible_input_shape(arg.get_shape(), expected);
                      }))
        MIGRAPHX_THROW("SELECT_MODULE: runtime input does not match symbolic input shape");
    std::unordered_map<sym::expr, std::size_t> values;
    if(not std::all_of(symbol_bindings.begin(), symbol_bindings.end(), [&](const auto& binding) {
           auto len               = args[binding.input_index].get_shape().lens().at(binding.axis);
           auto [found, inserted] = values.emplace(binding.symbol, len);
           return inserted or found->second == len;
       }))
        MIGRAPHX_THROW("SELECT_MODULE: repeated expression has inconsistent runtime dimensions");
    return values;
}

std::vector<shape> select_module::resolve_output_shapes(
    const std::unordered_map<sym::expr, std::size_t>& values) const
{
    if(input_shapes.empty())
        return {};
    std::vector<shape> result;
    std::transform(output_dyn_shapes.sub_shapes().begin(),
                   output_dyn_shapes.sub_shapes().end(),
                   std::back_inserter(result),
                   [&](const shape& output) { return resolve_output_shape(output, values); });
    return result;
}

std::vector<argument> select_module::reshape_outputs(
    const std::vector<argument>& results,
    const std::unordered_map<sym::expr, std::size_t>& values) const
{
    if(results.size() != num_outputs())
        MIGRAPHX_THROW("SELECT_MODULE: candidate returned the wrong number of outputs");
    if(input_shapes.empty())
        return results;
    std::vector<argument> reshaped;
    std::transform(results.begin(),
                   results.end(),
                   output_dyn_shapes.sub_shapes().begin(),
                   std::back_inserter(reshaped),
                   [&](const argument& result, const shape& output) {
                       auto resolved_output =
                           resolve_output_shape(output, values, &result.get_shape());
                       return reshape_output(result, resolved_output);
                   });
    return reshaped;
}

argument select_module::short_circuit_outputs(const std::vector<argument>& args,
                                              const std::vector<shape>& output_shapes) const
{
    std::vector<argument> results;
    results.reserve(num_outputs());
    if(not metadata.modules.front().outputs.empty())
    {
        std::transform(args.end() - num_outputs(),
                       args.end(),
                       output_shapes.begin(),
                       std::back_inserter(results),
                       &reshape_output);
    }
    else
    {
        std::transform(output_shapes.begin(),
                       output_shapes.end(),
                       std::back_inserter(results),
                       fix<argument>([](auto self, const shape& s) {
                           if(s.type() != shape::tuple_type)
                               return argument{s, nullptr};
                           std::vector<argument> sub_objects;
                           std::transform(
                               s.sub_shapes().begin(),
                               s.sub_shapes().end(),
                               std::back_inserter(sub_objects),
                               [&](const shape& sub_shape) { return self(sub_shape); });
                           return argument{sub_objects};
                       }));
    }
    return argument{results};
}

select_module::module_set_metadata
select_module::build_module_metadata(const std::vector<module_ref>& candidates) const
{
    module_set_metadata result;
    result.modules.reserve(candidates.size());
    auto candidate_indices = range(candidates.size());
    std::transform(
        candidates.begin(),
        candidates.end(),
        candidate_indices.begin(),
        std::back_inserter(result.modules),
        [&](module_ref mod, std::size_t index) { return make_module_metadata(*this, mod, index); });
    std::vector<std::vector<std::size_t>> selectors;
    selectors.reserve(result.modules.size());
    std::transform(result.modules.begin(),
                   result.modules.end(),
                   std::back_inserter(selectors),
                   [&](const module_metadata& candidate) {
                       return selector_indices(result.modules, candidate);
                   });
    migraphx::for_each(result.modules.begin(),
                       result.modules.end(),
                       selectors.begin(),
                       [](module_metadata& candidate, std::vector<std::size_t>& indices) {
                           candidate.selector_indices       = std::move(indices);
                           candidate.shared_dynamic_indices = shared_dynamic_indices(candidate);
                       });
    return result;
}

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
