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
using parameter_source   = select_module::parameter_source;
using source_kind        = select_module::source_kind;

std::size_t parameter_order(instruction_ref parameter)
{
    return any_cast<builtin::param>(parameter->get_operator()).order;
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
                               auto* output_shape = shape_at_path(output_shapes.at(r), output_path);
                               auto* leaf_shape   = shape_at_path(parameter_shape, leaf_path);
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
    auto slots = transform_accumulate(
        parameters.begin(),
        parameters.end(),
        std::size_t{0},
        [](std::size_t x, std::size_t y) { return std::max(x, y); },
        [](instruction_ref parameter) { return parameter_order(parameter) + 1; });
    result.parameters.resize(slots, parameter_source{source_kind::unused, 0});

    auto input_names = select.get_input_parameter_names(mod);
    std::transform(input_names.begin(),
                   input_names.end(),
                   std::back_inserter(result.inputs),
                   [&, index = std::size_t{0}](const std::string& name) mutable {
                       auto parameter = parameter_by_name.at(name);
                       result.parameters[parameter_order(parameter)] =
                           parameter_source{source_kind::input, index++};
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
        [&, index = std::size_t{0}](const std::string& name) mutable {
            auto parameter = parameter_by_name.at(name);
            result.parameters[parameter_order(parameter)] =
                parameter_source{source_kind::output, index++};
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

} // namespace

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
