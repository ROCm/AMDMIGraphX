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

#include <migraphx/split_prefill_decode.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/sym.hpp>
#include <algorithm>
#include <cstddef>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

namespace {

sym::expr sequence_length_symbol() { return sym::var("sequence_length"); }

struct sequence_dimension_info
{
    std::optional<shape::dynamic_dimension::interval> interval;
    bool valid = true;
};

std::optional<std::size_t>
find_max_sequence_length(const std::unordered_map<std::string, shape>& param_shapes)
{
    auto info = std::accumulate(
        param_shapes.begin(),
        param_shapes.end(),
        sequence_dimension_info{},
        [](auto result, const auto& param) {
            if(not param.second.symbolic())
                return result;
            return std::accumulate(
                param.second.dyn_dims().begin(),
                param.second.dyn_dims().end(),
                std::move(result),
                [](auto current, const auto& dd) {
                    if(not sym::same_symbol(dd.sym_expr, sequence_length_symbol()))
                        return current;
                    auto interval = dd.get_interval();
                    if(current.interval.has_value() and current.interval.value() != interval)
                        current.valid = false;
                    else
                        current.interval = interval;
                    return current;
                });
        });
    if(not info.valid or not info.interval.has_value() or info.interval->min != 1 or
       info.interval->max <= 1)
        return std::nullopt;

    return info.interval->max;
}

// Preserve existing parameter dispatch and avoid nesting select_module on repeated runs.
bool already_split(const_module_ref mod)
{
    auto param_names = mod->get_parameter_names();
    return std::any_of(param_names.begin(), param_names.end(), [&](const auto& param_name) {
        auto outputs = mod->get_parameter(param_name)->outputs();
        return std::any_of(outputs.begin(), outputs.end(), [](auto ins) {
            return ins->name() == "select_module";
        });
    });
}

shape specialize_sequence_length(const shape& s,
                                 const std::unordered_map<sym::expr, sym::expr>& bindings)
{
    if(not s.symbolic())
        return s;

    std::vector<shape::dynamic_dimension> dims(s.ndim());
    std::transform(s.dyn_dims().begin(), s.dyn_dims().end(), dims.begin(), [&](const auto& dd) {
        return shape::dynamic_dimension{dd.sym_expr.subs(bindings)};
    });
    std::vector<sym::expr> strides(s.dyn_strides().size());
    std::transform(s.dyn_strides().begin(),
                   s.dyn_strides().end(),
                   strides.begin(),
                   [&](const auto& stride) { return stride.subs(bindings); });

    shape result{s.type(), std::move(dims), std::move(strides)};
    if(result.is_fixed())
        return result.to_static(std::unordered_map<sym::expr, std::size_t>{});
    return result;
}

module_ref
create_specialized_module(module_pass_manager& mpm,
                          module_ref root,
                          const std::vector<std::pair<std::string, instruction_ref>>& parameters,
                          const std::string& name,
                          std::size_t sequence_length)
{
    auto* submod = mpm.create_module(name);
    std::unordered_map<instruction_ref, instruction_ref> map_ins;
    const std::unordered_map<sym::expr, sym::expr> bindings = {
        {sequence_length_symbol(), sym::lit(sequence_length)}};

    std::transform(parameters.begin(),
                   parameters.end(),
                   std::inserter(map_ins, map_ins.end()),
                   [&](const auto& parameter) {
                       const auto& [param_name, param] = parameter;
                       auto s = specialize_sequence_length(param->get_shape(), bindings);
                       return std::make_pair(param,
                                             submod->add_parameter(param_name, std::move(s)));
                   });

    // Keep literals in the root so both specializations share constants instead of copying them.
    auto instructions = iterator_for(*root);
    transform_if(
        instructions.begin(),
        instructions.end(),
        std::inserter(map_ins, map_ins.end()),
        [](auto ins) { return ins->name() == "@literal"; },
        [](auto literal) { return std::make_pair(literal, literal); });

    submod->add_return(submod->add_instructions(root, &map_ins));
    return submod;
}

} // namespace

void split_prefill_decode::apply(module_pass_manager& mpm) const
{
    auto* root = &mpm.get_module();
    if(root != mpm.get_root_module() or already_split(root))
        return;

    auto max_sequence_length = find_max_sequence_length(root->get_parameter_shapes());
    if(not max_sequence_length.has_value())
        return;

    auto param_names = root->get_parameter_names();
    std::sort(param_names.begin(), param_names.end());
    std::vector<std::pair<std::string, instruction_ref>> parameters(param_names.size());
    std::transform(
        param_names.begin(), param_names.end(), parameters.begin(), [&](const auto& param_name) {
            return std::make_pair(param_name, root->get_parameter(param_name));
        });
    const auto module_prefix           = root->name() + ":split_prefill_decode:";
    std::vector<module_ref> submodules = {
        create_specialized_module(mpm, root, parameters, module_prefix + "decode", 1),
        create_specialized_module(
            mpm, root, parameters, module_prefix + "prefill", max_sequence_length.value())};

    std::vector<instruction_ref> inputs;
    std::transform(parameters.begin(),
                   parameters.end(),
                   std::back_inserter(inputs),
                   [](const auto& parameter) { return parameter.second; });

    auto output_shapes = root->get_output_shapes();
    auto select        = root->add_instruction(
        make_op("select_module", {{"output_dyn_shapes", to_value(shape{output_shapes})}}),
        inputs,
        submodules);
    std::vector<instruction_ref> outputs(output_shapes.size());
    auto output_indices = migraphx::range(output_shapes.size());
    std::transform(
        output_indices.begin(), output_indices.end(), outputs.begin(), [&](std::size_t index) {
            return root->add_instruction(make_op("get_tuple_elem", {{"index", index}}), select);
        });
    root->replace_return(outputs);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
