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
#ifndef MIGRAPHX_GUARD_OPERATORS_SELECT_MODULE_HPP
#define MIGRAPHX_GUARD_OPERATORS_SELECT_MODULE_HPP

#include <migraphx/algorithm.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/context.hpp>
#include <migraphx/config.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <algorithm>
#include <cassert>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct select_module
{
    shape output_dyn_shapes;
    // An empty tensor at any listed input bypasses module dispatch.
    std::vector<std::size_t> empty_guard_inputs;

    struct output_location
    {
        std::size_t index;
        std::vector<std::size_t> path;
    };

    struct parameter_metadata
    {
        std::string name;
        shape parameter_shape;
        // Caller buffers this parameter writes, in parameter-leaf order. A location identifies
        // both the submodule return slot and a path within that return's buffer.
        std::vector<output_location> output_locations;
    };

    struct module_metadata
    {
        std::size_t module_index;
        std::vector<parameter_metadata> inputs;
        std::vector<parameter_metadata> outputs;
        std::vector<std::size_t> selector_indices;
        // Dynamic inputs that every candidate shares. Parameter evaluation does not check
        // dynamic shapes, so these are checked once the module is selected.
        std::vector<std::size_t> shared_dynamic_indices;
    };

    struct module_set_metadata
    {
        std::vector<module_metadata> modules;
    };

    // Built from the instruction's fixed submodule list during finalization. Candidate indices
    // avoid retaining non-owning module pointers when a finalized program is copied.
    module_set_metadata metadata;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_dyn_shapes, "output_dyn_shapes"),
                    f(self.empty_guard_inputs, "empty_guard_inputs"));
    }

    std::string name() const { return "select_module"; }

    std::size_t num_outputs() const { return output_dyn_shapes.sub_shapes().size(); }

    shape compute_shape(const std::vector<shape>& inputs,
                        const std::vector<module_ref>& submodules) const
    {
        check_shapes{inputs, *this, true}.has_at_least(1);
        if(any_of(empty_guard_inputs, [&](auto input) { return input >= inputs.size(); }))
            MIGRAPHX_THROW("SELECT_MODULE: empty guard input is out of range");
        // Once lowered, the output buffers follow the inputs and each input is bound to a
        // parameter of the compiled submodules, so a static input must keep that layout
        for(const auto& sub : submodules)
        {
            auto names = get_input_parameter_names(sub);
            if(inputs.size() <= names.size())
                continue;
            for(std::size_t i = 0; i < names.size(); i++)
            {
                auto expected = sub->get_parameter_shape(names[i]);
                if(inputs[i].dynamic() or expected.dynamic())
                    continue;
                if(not compatible_input_shape(inputs[i], expected))
                    MIGRAPHX_THROW("SELECT_MODULE: input " + std::to_string(i) +
                                   " does not have the layout of parameter " + names[i]);
            }
        }
        return shape{output_dyn_shapes};
    }

    void finalize(context&,
                  const shape&,
                  const std::vector<shape>&,
                  const std::vector<module_ref>& submodule_list)
    {
        metadata = build_module_metadata(submodule_list);
    }

    std::vector<std::string> get_input_parameter_names(const_module_ref mod) const
    {
        auto param_names = mod->get_parameter_names();
        std::vector<std::string> ret;
        std::copy_if(param_names.cbegin(),
                     param_names.cend(),
                     std::back_inserter(ret),
                     [](const auto& pn) { return not contains(pn, "#output_"); });
        std::sort(ret.begin(), ret.end());
        return ret;
    }

    std::vector<std::string> get_output_parameter_names(const_module_ref mod) const
    {
        auto param_names = mod->get_parameter_names();
        std::vector<std::string> ret;
        std::copy_if(param_names.cbegin(),
                     param_names.cend(),
                     std::back_inserter(ret),
                     [](const auto& pn) { return contains(pn, "#output_"); });
        // needs to be sorted to ensure output parameter ordering
        std::sort(ret.begin(), ret.end());
        return ret;
    }

    static bool compatible_input_shape(const shape& actual, const shape& expected)
    {
        if(actual == expected)
            return true;
        if(expected.dynamic())
            return actual.type() == expected.type() and shape::is_compatible_lens(actual, expected);
        if(actual.type() != expected.type() or actual.type() == shape::tuple_type or
           not shape::same_lens(actual, expected))
            return false;
        if(actual.elements() == 0)
            return true;
        return all_of(range(actual.ndim()), [&](auto axis) {
            return actual.lens().at(axis) <= 1 or
                   actual.strides().at(axis) == expected.strides().at(axis);
        });
    }

    MIGRAPHX_EXPORT module_set_metadata
    build_module_metadata(const std::vector<module_ref>& candidates) const;

    const module_set_metadata& finalized_metadata(std::size_t module_count) const
    {
        if(metadata.modules.empty() or metadata.modules.size() != module_count)
            MIGRAPHX_THROW("SELECT_MODULE: operation was not finalized");
        return metadata;
    }

    bool should_short_circuit(const std::vector<argument>& args) const
    {
        return any_of(empty_guard_inputs, [&](auto input) {
            if(input >= args.size())
                MIGRAPHX_THROW("SELECT_MODULE: empty guard input is out of range");
            return args.at(input).get_shape().elements() == 0;
        });
    }

    argument short_circuit_outputs(const module_set_metadata& finalized,
                                   const std::vector<argument>& args) const
    {
        std::vector<argument> results;
        results.reserve(num_outputs());
        if(not finalized.modules.front().outputs.empty())
        {
            if(args.size() < num_outputs())
                MIGRAPHX_THROW("SELECT_MODULE: missing output allocations");
            std::copy(args.end() - num_outputs(), args.end(), std::back_inserter(results));
        }
        else
        {
            auto make_max_argument = fix<argument>([](auto self, const shape& s) {
                if(s.type() != shape::tuple_type)
                    return argument{shape{s.type(), s.max_lens()}};
                std::vector<argument> sub_objects;
                sub_objects.reserve(s.sub_shapes().size());
                std::transform(s.sub_shapes().begin(),
                               s.sub_shapes().end(),
                               std::back_inserter(sub_objects),
                               [&](const shape& sub_shape) { return self(sub_shape); });
                return argument{sub_objects};
            });
            std::transform(output_dyn_shapes.sub_shapes().begin(),
                           output_dyn_shapes.sub_shapes().end(),
                           std::back_inserter(results),
                           make_max_argument);
        }
        return argument{results};
    }

    // Input arguments are ordered like the sorted input parameters, followed by one buffer per
    // output when the submodules have output parameters. Only the positions whose parameter
    // differs between candidates are compared to pick a module. A shared input that does not match
    // would reject every candidate, so the shared dynamic inputs are only checked on the selected
    // module; the static ones are checked when the submodule evaluates its parameters.
    template <class GetArgument>
    const module_metadata& find_module(const module_set_metadata& modules_metadata,
                                       std::size_t argument_count,
                                       GetArgument get_argument) const
    {
        auto module_iter =
            std::find_if(modules_metadata.modules.begin(),
                         modules_metadata.modules.end(),
                         [&](const auto& info) {
                             return info.inputs.size() <= argument_count and
                                    std::all_of(info.selector_indices.begin(),
                                                info.selector_indices.end(),
                                                [&](std::size_t index) {
                                                    return compatible_input_shape(
                                                        get_argument(index).get_shape(),
                                                        info.inputs[index].parameter_shape);
                                                });
                         });

        if(module_iter == modules_metadata.modules.end() or
           not std::all_of(module_iter->shared_dynamic_indices.begin(),
                           module_iter->shared_dynamic_indices.end(),
                           [&](std::size_t index) {
                               return compatible_input_shape(
                                   get_argument(index).get_shape(),
                                   module_iter->inputs[index].parameter_shape);
                           }))
        {
            MIGRAPHX_THROW("SELECT_MODULE: no compatible submodules found for given input shapes");
        }
        if(not module_iter->outputs.empty() and
           argument_count != module_iter->inputs.size() + num_outputs())
            MIGRAPHX_THROW("SELECT_MODULE: missing output allocations");
        return *module_iter;
    }

    argument prepare_output_shape(const parameter_metadata& output,
                                  const shape& expected,
                                  const argument& arg) const
    {
        if(arg.get_shape() == expected)
            return arg;
        // Reshaping onto a smaller buffer would let the submodule write past its end, so refuse
        // rather than corrupt memory.
        if(arg.get_shape().bytes() < expected.bytes())
            MIGRAPHX_THROW("SELECT_MODULE: output buffer for \"" + output.name + "\" holds " +
                           std::to_string(arg.get_shape().bytes()) + " bytes but the selected " +
                           "submodule writes " + std::to_string(expected.bytes()));
        return arg.reshape(expected);
    }

    // get_output returns the caller's buffer for a return of the submodule.
    template <class GetOutput>
    argument prepare_output(const parameter_metadata& output, GetOutput get_output) const
    {
        std::size_t leaf_index = 0;
        auto prepared          = fix<argument>([&](auto self, const shape& expected) {
            if(expected.type() != shape::tuple_type)
            {
                const auto& location = output.output_locations.at(leaf_index++);
                auto result          = get_output(location.index);
                for(auto index : location.path)
                    result = result.get_sub_object(index);
                return prepare_output_shape(output, expected, result);
            }

            std::vector<argument> result;
            result.reserve(expected.sub_shapes().size());
            std::transform(expected.sub_shapes().begin(),
                           expected.sub_shapes().end(),
                           std::back_inserter(result),
                           [&](const shape& sub_shape) { return self(sub_shape); });
            return argument{result};
        })(output.parameter_shape);
        assert(leaf_index == output.output_locations.size());
        return prepared;
    }

    argument compute(const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& submodule_list,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        const auto& finalized = finalized_metadata(submodule_list.size());
        if(should_short_circuit(args))
            return short_circuit_outputs(finalized, args);
        const auto& module_info =
            find_module(finalized, args.size(), [&](std::size_t index) -> const argument& {
                return args[index];
            });
        auto* module_to_run = submodule_list.at(module_info.module_index);
        std::unordered_map<std::string, argument> p_map;
        p_map.reserve(module_info.inputs.size() + module_info.outputs.size());

        // add input parameters to parameter_map
        std::transform(module_info.inputs.begin(),
                       module_info.inputs.end(),
                       args.begin(),
                       std::inserter(p_map, p_map.end()),
                       [](const auto& input, const auto& arg) {
                           const auto& expected = input.parameter_shape;
                           if(not expected.dynamic() and arg.get_shape() != expected)
                           {
                               if(not compatible_input_shape(arg.get_shape(), expected))
                                   MIGRAPHX_THROW(
                                       "SELECT_MODULE: no compatible submodules found for given "
                                       "input shapes");
                               return std::make_pair(input.name, arg.reshape(expected));
                           }
                           return std::make_pair(input.name, arg);
                       });

        // Each output of the submodule writes into the caller's buffer for that output. A compiled
        // output parameter can itself be a tuple when one kernel produces multiple returns.
        auto output_start = args.size() - num_outputs();
        auto get_output   = [&](std::size_t index) { return args[output_start + index]; };
        std::transform(module_info.outputs.begin(),
                       module_info.outputs.end(),
                       std::inserter(p_map, p_map.end()),
                       [&](const auto& output) {
                           return std::make_pair(output.name, prepare_output(output, get_output));
                       });
        auto results = run(module_to_run, p_map);
        return argument{results};
    }

    // The caller's output buffers are appended after the input arguments during lowering.
    std::vector<std::size_t> output_alias(const std::vector<shape>& shapes) const
    {
        if(shapes.size() <= num_outputs())
            return {};
        std::vector<std::size_t> result(num_outputs());
        std::iota(result.begin(), result.end(), shapes.size() - num_outputs());
        return result;
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
