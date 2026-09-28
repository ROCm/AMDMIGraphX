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

// The element of a tuple output parameter that a return writes, found by following the return's
// aliases back to the get_tuple_elem that reads the parameter.
std::optional<std::size_t> returned_tuple_element(instruction_ref ret, instruction_ref parameter)
{
    return fix<std::optional<std::size_t>>([&](auto self, instruction_ref ins) {
        if(ins == parameter)
            return std::optional<std::size_t>{};
        if(auto index = tuple_element_index(ins))
            return index;
        auto aliases = ins->get_operator().output_alias(to_shapes(ins->inputs()));
        if(aliases.empty())
            return std::optional<std::size_t>{};
        return self(ins->inputs().at(aliases.front()));
    })(ret);
}

bool aliases_parameter(instruction_ref ret, instruction_ref parameter)
{
    return contains(instruction::get_output_alias(ret), parameter) or
           contains(ret->inputs(), parameter);
}

// The return slots an output parameter writes. A fused kernel can pack several returns into one
// tuple parameter, whose elements need not be consecutive returns.
std::vector<std::size_t> output_indices(const std::string& name,
                                        instruction_ref parameter,
                                        const std::vector<instruction_ref>& returns)
{
    const auto& parameter_shape = parameter->get_shape();
    if(parameter_shape.type() != shape::tuple_type)
    {
        auto ret = std::find_if(returns.begin(), returns.end(), [&](instruction_ref r) {
            return aliases_parameter(r, parameter);
        });
        if(ret == returns.end())
            MIGRAPHX_THROW("SELECT_MODULE: output parameter \"" + name +
                           "\" does not alias a module output");
        return {static_cast<std::size_t>(std::distance(returns.begin(), ret))};
    }

    constexpr auto missing = std::numeric_limits<std::size_t>::max();
    std::vector<std::size_t> result(parameter_shape.sub_shapes().size(), missing);
    auto return_indices = range(returns.size());
    migraphx::for_each(returns.begin(),
                       returns.end(),
                       return_indices.begin(),
                       [&](instruction_ref ret, std::size_t r) {
                           if(not aliases_parameter(ret, parameter))
                               return;
                           auto element = returned_tuple_element(ret, parameter);
                           if(element.has_value() and *element < result.size())
                               result[*element] = r;
                       });
    if(contains(result, missing))
        MIGRAPHX_THROW("SELECT_MODULE: tuple output parameter \"" + name +
                       "\" is not aliased by get_tuple_elem returns for every subobject");
    return result;
}

module_metadata make_module_metadata(const select_module& select, module_ref mod)
{
    module_metadata result;
    result.mod      = mod;
    auto parameters = mod->get_parameters();
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
    std::transform(output_names.begin(),
                   output_names.end(),
                   std::back_inserter(result.outputs),
                   [&, index = std::size_t{0}](const std::string& name) mutable {
                       auto parameter = parameter_by_name.at(name);
                       result.parameters[parameter_order(parameter)] =
                           parameter_source{source_kind::output, index++};
                       return parameter_metadata{
                           name, parameter->get_shape(), output_indices(name, parameter, returns)};
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

} // namespace

select_module::module_set_metadata
select_module::build_module_metadata(const std::vector<module_ref>& candidates) const
{
    module_set_metadata result;
    result.candidates = candidates;
    result.modules.reserve(candidates.size());
    std::transform(candidates.begin(),
                   candidates.end(),
                   std::back_inserter(result.modules),
                   [&](module_ref mod) { return make_module_metadata(*this, mod); });
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
                           candidate.selector_indices = std::move(indices);
                       });
    return result;
}

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
