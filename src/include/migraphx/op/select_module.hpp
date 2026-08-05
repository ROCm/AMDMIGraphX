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
#include <memory>
#include <mutex>
#include <numeric>

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
    };

    struct module_metadata
    {
        module_ref mod;
        std::vector<parameter_metadata> inputs;
        std::vector<parameter_metadata> outputs;
        std::vector<std::size_t> selector_indices;
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
        std::vector<cache_entry> entries;
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
        std::lock_guard<std::mutex> lock{cache->mutex};
        auto entry = std::find_if(cache->entries.begin(), cache->entries.end(), [&](const auto& e) {
            return e.modules == submodule_list;
        });
        if(entry != cache->entries.end())
            return entry->metadata;

        auto metadata = std::make_shared<module_set_metadata>();
        metadata->modules.reserve(submodule_list.size());
        std::transform(
            submodule_list.begin(),
            submodule_list.end(),
            std::back_inserter(metadata->modules),
            [&](module_ref mod) {
                module_metadata result;
                result.mod        = mod;
                auto param_shapes = mod->get_parameter_shapes();
                auto add_metadata = [&](const auto& names, auto output) {
                    std::transform(names.begin(), names.end(), output, [&](const auto& name) {
                        return parameter_metadata{name, param_shapes.at(name)};
                    });
                };
                add_metadata(get_input_parameter_names(mod), std::back_inserter(result.inputs));
                add_metadata(get_output_parameter_names(mod), std::back_inserter(result.outputs));
                return result;
            });
        std::vector<std::vector<std::size_t>> selectors;
        std::transform(metadata->modules.begin(),
                       metadata->modules.end(),
                       std::back_inserter(selectors),
                       [&](const auto& candidate) {
                           std::vector<std::size_t> indices(candidate.inputs.size());
                           std::iota(indices.begin(), indices.end(), 0);
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
        std::vector<std::size_t> module_indices(metadata->modules.size());
        std::iota(module_indices.begin(), module_indices.end(), 0);
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
        cache->entries.push_back({submodule_list, metadata});
        return metadata;
    }

    argument compute(const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& submodule_list,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        // Find the submodule from the input positions whose parameter shapes differ between
        // candidates. The selected submodule still validates every parameter during evaluation.
        auto metadata = get_module_metadata(submodule_list);
        auto module_iter =
            std::find_if(metadata->modules.begin(), metadata->modules.end(), [&](const auto& info) {
                assert(info.inputs.size() <= args.size());
                return std::all_of(info.selector_indices.begin(),
                                   info.selector_indices.end(),
                                   [&](std::size_t index) {
                                       return index < info.inputs.size() and index < args.size() and
                                              args[index].get_shape() ==
                                                  info.inputs[index].parameter_shape;
                                   });
            });

        if(module_iter == metadata->modules.end())
        {
            MIGRAPHX_THROW("SELECT_MODULE: no compatible submodules found for given input shapes");
        }

        const auto& module_info = *module_iter;
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

        // One tuple output parameter in main module to multiple output parameters in submodule
        auto output_sub_objects = args.back().get_sub_objects();
        assert(module_info.outputs.size() == output_sub_objects.size());
        std::transform(module_info.outputs.begin(),
                       module_info.outputs.end(),
                       output_sub_objects.begin(),
                       std::inserter(p_map, p_map.end()),
                       [&](const auto& output, const auto& a) {
                           const auto& name = output.name;
                           const auto& ps   = output.parameter_shape;
                           if(a.get_shape() == ps)
                               return std::make_pair(name, a);
                           // Reshaping onto a smaller buffer would let the submodule write past
                           // its end, so refuse rather than corrupt memory.
                           if(a.get_shape().bytes() < ps.bytes())
                               MIGRAPHX_THROW("SELECT_MODULE: output buffer for \"" + name +
                                              "\" holds " + std::to_string(a.get_shape().bytes()) +
                                              " bytes but the selected submodule writes " +
                                              std::to_string(ps.bytes()));
                           return std::make_pair(name, a.reshape(ps));
                       });
        auto results = run(module_to_run, p_map);
        return argument{results};
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
