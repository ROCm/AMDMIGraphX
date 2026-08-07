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
#include <migraphx/module.hpp>
#include <migraphx/algorithm.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/ranges.hpp>
#include <memory>
#include <mutex>

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
        std::size_t output_offset = 0;
        std::size_t output_count  = 1;
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
        bool leaf_captures = true;
    };

    struct module_set_metadata
    {
        std::vector<module_metadata> modules;
    };

    struct cache_entry
    {
        std::vector<module_ref> modules;
        std::shared_ptr<const module_set_metadata> metadata;
    };

    struct metadata_cache
    {
        std::mutex mutex;
        std::vector<std::shared_ptr<const cache_entry>> entries;
        std::shared_ptr<const cache_entry> last_entry;
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

    std::shared_ptr<const module_set_metadata>
    get_module_metadata(const std::vector<module_ref>& submodule_list) const
    {
        auto last_entry = std::atomic_load(&cache->last_entry);
        if(last_entry != nullptr and last_entry->modules == submodule_list)
            return last_entry->metadata;

        std::lock_guard<std::mutex> lock{cache->mutex};
        auto entry = std::find_if(cache->entries.begin(), cache->entries.end(), [&](const auto& e) {
            return e->modules == submodule_list;
        });
        if(entry != cache->entries.end())
        {
            std::atomic_store(&cache->last_entry, *entry);
            return (*entry)->metadata;
        }

        auto metadata = std::make_shared<module_set_metadata>();
        metadata->modules.reserve(submodule_list.size());
        std::transform(
            submodule_list.begin(),
            submodule_list.end(),
            std::back_inserter(metadata->modules),
            [&](module_ref mod) {
                module_metadata result;
                result.mod   = mod;
                auto modules = mod->get_sub_modules();
                modules.push_back(mod);
                result.leaf_captures =
                    std::all_of(modules.begin(), modules.end(), [](module_ref current) {
                        return std::all_of(current->begin(), current->end(), [&](const auto& ins) {
                            return std::all_of(ins.inputs().begin(),
                                               ins.inputs().end(),
                                               [&](instruction_ref input) {
                                                   return current->has_instruction(input) or
                                                          (input->inputs().empty() and
                                                           input->module_inputs().empty());
                                               });
                        });
                    });
                auto param_shapes = mod->get_parameter_shapes();
                auto parameters   = mod->get_parameters();
                std::unordered_map<std::string, std::size_t> param_orders;
                param_orders.reserve(parameters.size());
                std::transform(parameters.begin(),
                               parameters.end(),
                               std::inserter(param_orders, param_orders.end()),
                               [](instruction_ref ins) {
                                   const auto& param =
                                       any_cast<builtin::param>(ins->get_operator());
                                   return std::make_pair(param.parameter, std::size_t{param.order});
                               });
                std::size_t parameter_slots = 0;
                if(not param_orders.empty())
                {
                    auto max_order = std::max_element(
                        param_orders.begin(), param_orders.end(), [](const auto& x, const auto& y) {
                            return x.second < y.second;
                        });
                    parameter_slots = max_order->second + 1;
                }
                result.parameters.resize(parameter_slots, parameter_source{source_kind::unused, 0});
                auto input_names = get_input_parameter_names(mod);
                result.inputs.reserve(input_names.size());
                std::transform(input_names.begin(),
                               input_names.end(),
                               std::back_inserter(result.inputs),
                               [&, index = std::size_t{0}](const auto& name) mutable {
                                   auto order = param_orders.at(name);
                                   result.parameters[order] =
                                       parameter_source{source_kind::input, index++};
                                   return parameter_metadata{name, param_shapes.at(name)};
                               });

                auto output_names = get_output_parameter_names(mod);
                auto returns      = mod->get_returns();
                result.outputs.reserve(output_names.size());
                std::transform(
                    output_names.begin(),
                    output_names.end(),
                    std::back_inserter(result.outputs),
                    [&, index = std::size_t{0}, offset = std::size_t{0}](const auto& name) mutable {
                        auto parameter = std::find_if(
                            parameters.begin(), parameters.end(), [&](instruction_ref ins) {
                                return any_cast<builtin::param>(ins->get_operator()).parameter ==
                                       name;
                            });
                        assert(parameter != parameters.end());
                        auto output_count = transform_accumulate(
                            returns.begin(),
                            returns.end(),
                            std::size_t{0},
                            std::plus<>{},
                            [&](instruction_ref ret) {
                                auto aliases = instruction::get_output_alias(ret);
                                return std::size_t{contains(aliases, *parameter)};
                            });
                        if(output_count == 0)
                            output_count = 1;

                        auto order               = param_orders.at(name);
                        result.parameters[order] = parameter_source{source_kind::output, index++};
                        parameter_metadata output{
                            name, param_shapes.at(name), offset, output_count};
                        offset += output_count;
                        return output;
                    });
                return result;
            });
        std::vector<std::vector<std::size_t>> selectors;
        std::transform(metadata->modules.begin(),
                       metadata->modules.end(),
                       std::back_inserter(selectors),
                       [&](const auto& candidate) {
                           auto indices = range(candidate.inputs.size());
                           std::vector<std::size_t> result;
                           std::copy_if(indices.begin(),
                                        indices.end(),
                                        std::back_inserter(result),
                                        [&](std::size_t index) {
                                            const auto& expected = candidate.inputs[index];
                                            return std::any_of(
                                                metadata->modules.begin(),
                                                metadata->modules.end(),
                                                [&](const auto& info) {
                                                    if(index >= info.inputs.size())
                                                        return true;
                                                    const auto& input = info.inputs[index];
                                                    return input.name != expected.name or
                                                           input.parameter_shape !=
                                                               expected.parameter_shape;
                                                });
                                        });
                           return result;
                       });
        auto module_indices = range(metadata->modules.size());
        std::vector<module_metadata> modules;
        modules.reserve(metadata->modules.size());
        std::transform(module_indices.begin(),
                       module_indices.end(),
                       std::back_inserter(modules),
                       [&](std::size_t index) {
                           auto result             = std::move(metadata->modules[index]);
                           result.selector_indices = std::move(selectors[index]);
                           return result;
                       });
        metadata->modules = std::move(modules);
        auto new_entry =
            std::make_shared<const cache_entry>(cache_entry{submodule_list, std::move(metadata)});
        cache->entries.push_back(new_entry);
        std::atomic_store(&cache->last_entry, new_entry);
        return new_entry->metadata;
    }

    bool has_only_leaf_captures(const std::vector<module_ref>& submodule_list) const
    {
        auto metadata = get_module_metadata(submodule_list);
        return std::all_of(metadata->modules.begin(),
                           metadata->modules.end(),
                           [](const auto& info) { return info.leaf_captures; });
    }

    template <class GetArgument>
    const module_metadata& find_module(const std::shared_ptr<const module_set_metadata>& metadata,
                                       std::size_t argument_count,
                                       GetArgument get_argument) const
    {
        auto module_iter =
            std::find_if(metadata->modules.begin(), metadata->modules.end(), [&](const auto& info) {
                assert(info.inputs.size() <= argument_count);
                return std::all_of(info.selector_indices.begin(),
                                   info.selector_indices.end(),
                                   [&](std::size_t index) {
                                       return index < info.inputs.size() and
                                              index < argument_count and
                                              get_argument(index).get_shape() ==
                                                  info.inputs[index].parameter_shape;
                                   });
            });

        if(module_iter == metadata->modules.end())
        {
            MIGRAPHX_THROW("SELECT_MODULE: no compatible submodules found for given input shapes");
        }
        return *module_iter;
    }

    const module_metadata& find_module(const std::shared_ptr<const module_set_metadata>& metadata,
                                       const std::vector<argument>& args) const
    {
        return find_module(metadata, args.size(), [&](std::size_t index) -> const argument& {
            return args[index];
        });
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
        if(output.output_offset + output.output_count > output_shapes.size())
            MIGRAPHX_THROW("SELECT_MODULE: selected submodule needs more output buffers than the "
                           "main module provides");

        if(output.output_count == 1)
            return prepare_output_shape(
                output, output.parameter_shape, outputs.get_sub_object(output.output_offset));

        const auto& parameter_shapes = output.parameter_shape.sub_shapes();
        if(output.parameter_shape.type() != shape::tuple_type or
           parameter_shapes.size() != output.output_count)
            MIGRAPHX_THROW("SELECT_MODULE: tuple output parameter \"" + output.name +
                           "\" does not match the selected submodule returns");

        auto indices = range(output.output_count);
        std::vector<argument> result;
        result.reserve(output.output_count);
        std::transform(
            indices.begin(), indices.end(), std::back_inserter(result), [&](std::size_t index) {
                return prepare_output_shape(output,
                                            parameter_shapes[index],
                                            outputs.get_sub_object(output.output_offset + index));
            });
        return argument{result};
    }

    argument compute(const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& submodule_list,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        // Find the submodule from the input positions whose parameter shapes differ between
        // candidates. The selected submodule still validates every parameter during evaluation.
        auto metadata           = get_module_metadata(submodule_list);
        const auto& module_info = find_module(metadata, args);
        auto module_to_run      = module_info.mod;
        std::unordered_map<std::string, argument> p_map;
        p_map.reserve(module_info.inputs.size() + module_info.outputs.size());

        // add input parameters to parameter_map
        assert(module_info.inputs.size() <= args.size());
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
        const auto& module_info = find_module(metadata, argument_count, get_argument);
        assert(argument_count > 0);
        assert(module_info.inputs.size() + 1 == argument_count);
        auto params = positional_parameter_view<GetArgument>{
            this, &module_info, get_argument, get_argument(argument_count - 1)};
        auto module_to_run = module_info.mod;
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
