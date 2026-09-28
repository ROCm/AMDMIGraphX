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

#include <migraphx/check_shapes.hpp>
#include <migraphx/config.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct select_module
{
    shape output_dyn_shapes;

    struct parameter_metadata
    {
        std::string name;
        shape parameter_shape;
        // Parent tuple slots this parameter writes, in parameter-subobject order. A fused
        // GPU kernel can pack many returns into one tuple parameter whose get_tuple_elem
        // users are not consecutive in the submodule return list.
        std::vector<std::size_t> output_indices;
    };

    enum class source_kind
    {
        unused,
        input,
        output
    };

    struct parameter_source
    {
        source_kind kind;
        std::size_t index;
    };

    struct module_metadata
    {
        module_ref mod;
        std::vector<parameter_metadata> inputs;
        std::vector<parameter_metadata> outputs;
        std::vector<std::size_t> selector_indices;
        std::vector<parameter_source> parameters;
    };

    struct module_set_metadata
    {
        std::vector<module_ref> candidates;
        std::vector<module_metadata> modules;
    };

    struct metadata_cache
    {
        std::mutex mutex;
        std::vector<std::shared_ptr<const module_set_metadata>> entries;
        std::shared_ptr<const module_set_metadata> last_entry;
    };

    mutable std::shared_ptr<metadata_cache> cache = std::make_shared<metadata_cache>();

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_dyn_shapes, "output_dyn_shapes"));
    }

    std::string name() const { return "select_module"; }

    shape compute_shape(const std::vector<shape>& inputs, const std::vector<module_ref>&) const
    {
        check_shapes{inputs, *this, true}.has_at_least(1);
        return shape{output_dyn_shapes};
    }

    std::vector<std::string> get_input_parameter_names(module_ref mod) const
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

    std::vector<std::string> get_output_parameter_names(module_ref mod) const
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

    MIGRAPHX_EXPORT module_set_metadata
    build_module_metadata(const std::vector<module_ref>& candidates) const;

    // Built once for each list of candidates and shared by copies of the operator. Evaluation
    // almost always repeats the last list, which is checked without taking the lock.
    std::shared_ptr<const module_set_metadata>
    get_module_metadata(const std::vector<module_ref>& submodule_list) const
    {
        auto last_entry = std::atomic_load(&cache->last_entry);
        if(last_entry != nullptr and last_entry->candidates == submodule_list)
            return last_entry;

        std::lock_guard<std::mutex> lock{cache->mutex};
        auto entry = std::find_if(cache->entries.begin(), cache->entries.end(), [&](const auto& e) {
            return e->candidates == submodule_list;
        });
        if(entry == cache->entries.end())
            entry = cache->entries.insert(
                cache->entries.end(),
                std::make_shared<const module_set_metadata>(build_module_metadata(submodule_list)));
        std::atomic_store(&cache->last_entry, *entry);
        return *entry;
    }

    static bool matches_input_shape(const shape& actual, const shape& expected)
    {
        if(expected.dynamic())
            return actual.type() == expected.type() and shape::is_compatible_lens(actual, expected);
        return actual == expected;
    }

    // Input arguments are ordered like the sorted input parameters, followed by the tuple of
    // output buffers when the submodules have output parameters. Only the positions whose
    // parameter differs between candidates are compared; the selected submodule still validates
    // every parameter during evaluation.
    template <class GetArgument>
    const module_metadata& find_module(const module_set_metadata& metadata,
                                       std::size_t argument_count,
                                       GetArgument get_argument) const
    {
        auto module_iter =
            std::find_if(metadata.modules.begin(), metadata.modules.end(), [&](const auto& info) {
                return info.inputs.size() <= argument_count and
                       std::all_of(info.selector_indices.begin(),
                                   info.selector_indices.end(),
                                   [&](std::size_t index) {
                                       return matches_input_shape(
                                           get_argument(index).get_shape(),
                                           info.inputs[index].parameter_shape);
                                   });
            });

        if(module_iter == metadata.modules.end())
        {
            MIGRAPHX_THROW("SELECT_MODULE: no compatible submodules found for given input shapes");
        }
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

    argument prepare_output(const parameter_metadata& output, const argument& outputs) const
    {
        const auto& output_shapes = outputs.get_shape().sub_shapes();
        if(std::any_of(output.output_indices.begin(),
                       output.output_indices.end(),
                       [&](std::size_t index) { return index >= output_shapes.size(); }))
            MIGRAPHX_THROW("SELECT_MODULE: selected submodule needs more output buffers than the "
                           "main module provides");

        if(output.parameter_shape.type() != shape::tuple_type)
            return prepare_output_shape(output,
                                        output.parameter_shape,
                                        outputs.get_sub_object(output.output_indices.front()));

        const auto& parameter_shapes = output.parameter_shape.sub_shapes();
        std::vector<argument> result;
        result.reserve(parameter_shapes.size());
        std::transform(parameter_shapes.begin(),
                       parameter_shapes.end(),
                       output.output_indices.begin(),
                       std::back_inserter(result),
                       [&](const shape& expected, std::size_t index) {
                           return prepare_output_shape(
                               output, expected, outputs.get_sub_object(index));
                       });
        return argument{result};
    }

    argument compute(const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& submodule_list,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        auto metadata = get_module_metadata(submodule_list);
        const auto& module_info =
            find_module(*metadata, args.size(), [&](std::size_t index) -> const argument& {
                return args[index];
            });
        auto* module_to_run = module_info.mod;
        std::unordered_map<std::string, argument> p_map;
        p_map.reserve(module_info.inputs.size() + module_info.outputs.size());

        // add input parameters to parameter_map
        std::transform(
            module_info.inputs.begin(),
            module_info.inputs.end(),
            args.begin(),
            std::inserter(p_map, p_map.end()),
            [](const auto& input, const auto& arg) { return std::make_pair(input.name, arg); });

        // Route the main module's tuple of output buffers to the selected submodule. A compiled
        // output parameter can itself be a tuple when one kernel produces multiple returns.
        std::transform(module_info.outputs.begin(),
                       module_info.outputs.end(),
                       std::inserter(p_map, p_map.end()),
                       [&](const auto& output) {
                           return std::make_pair(output.name, prepare_output(output, args.back()));
                       });
        auto results = run(module_to_run, p_map);
        return argument{results};
    }

    template <class GetArgument>
    struct positional_parameter_view
    {
        const select_module* select;
        const module_metadata* metadata;
        GetArgument get_argument;
        argument output;

        argument get_parameter(std::size_t order) const
        {
            const auto& source = metadata->parameters.at(order);
            assert(source.kind != source_kind::unused);
            if(source.kind == source_kind::input)
                return get_argument(source.index);
            return select->prepare_output(metadata->outputs[source.index], output);
        }
    };

    template <class GetArgument, class Run>
    argument compute_with_positional_parameters(std::size_t argument_count,
                                                GetArgument get_argument,
                                                const std::vector<module_ref>& submodule_list,
                                                Run run) const
    {
        auto metadata           = get_module_metadata(submodule_list);
        const auto& module_info = find_module(*metadata, argument_count, get_argument);
        assert(argument_count > 0);
        auto params = positional_parameter_view<GetArgument>{
            this, &module_info, get_argument, get_argument(argument_count - 1)};
        auto* module_to_run = module_info.mod;
        return argument{run(module_to_run, params)};
    }

    std::vector<std::size_t> output_alias(const std::vector<shape>& shapes) const
    {
        return {shapes.size() - 1};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
