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
#ifndef MIGRAPHX_GUARD_OPERATORS_SELECT_MODULE_INDEX_HPP
#define MIGRAPHX_GUARD_OPERATORS_SELECT_MODULE_INDEX_HPP

#include <migraphx/check_shapes.hpp>
#include <migraphx/module.hpp>
#include <migraphx/stringutils.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/// Runs one of its submodules, chosen at runtime by an integer index.
///
/// select_module_index(index, data..., output) [submodules...]
///   index:   single-element integral scalar, read on the host
///   data:    passed to the chosen submodule as its parameters, in sorted name order
///   output:  tuple buffer the chosen submodule writes into (added by gpu lowering)
///
/// index_map gives the index value that selects each submodule: submodule i runs
/// when index == index_map[i]. For example, index_map = {4, 8} runs the first
/// submodule for index 4 and the second for index 8; any other index throws.
/// When index_map is empty, the index is the submodule's position, so index 0
/// runs the first submodule.
struct select_module_index
{
    std::vector<std::size_t> index_map;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.index_map, "index_map"));
    }

    std::string name() const { return "select_module_index"; }

    // Lowering appends the tuple output buffer as the last input
    static bool has_output_buffer(const std::vector<shape>& inputs)
    {
        return inputs.size() > 1 and inputs.back().type() == shape::tuple_type;
    }

    shape compute_shape(const std::vector<shape>& inputs, const std::vector<module_ref>& mods) const
    {
        check_shapes{inputs, *this}.has_at_least(1);
        const auto& index = inputs.front();
        // On GPU the index is a hip::load_scalar result; under ref it is a host integer.
        if(index.elements() != 1 or not shape::is_integral(index.type()))
        {
            MIGRAPHX_THROW("SELECT_MODULE_INDEX: index must be a static single-element "
                           "integral scalar.");
        }
        if(mods.empty())
        {
            MIGRAPHX_THROW("SELECT_MODULE_INDEX: operator requires at least one submodule.");
        }
        if(not index_map.empty() and index_map.size() != mods.size())
        {
            MIGRAPHX_THROW("SELECT_MODULE_INDEX: index_map must match submodule count.");
        }
        auto sorted_map = index_map;
        std::sort(sorted_map.begin(), sorted_map.end());
        auto duplicate = std::adjacent_find(sorted_map.begin(), sorted_map.end());
        if(duplicate != sorted_map.end())
        {
            MIGRAPHX_THROW("SELECT_MODULE_INDEX: index_map has duplicate entry " +
                           std::to_string(*duplicate) + ".");
        }

        auto out_shapes0 = mods.front()->get_output_shapes();
        if(std::any_of(mods.begin() + 1, mods.end(), [&](module_ref mod) {
               return mod->get_output_shapes() != out_shapes0;
           }))
        {
            MIGRAPHX_THROW("SELECT_MODULE_INDEX: output shapes of submodules must be the same.");
        }
        shape output{out_shapes0};

        std::vector<shape> data_shapes(inputs.begin() + 1,
                                       has_output_buffer(inputs) ? inputs.end() - 1 : inputs.end());
        // A trailing tuple is taken as the output buffer, so a tuple data input
        // would be ambiguous with it
        if(std::any_of(data_shapes.begin(),
                       data_shapes.end(),
                       [](const shape& s) { return s.type() == shape::tuple_type; }) or
           (has_output_buffer(inputs) and not shape::is_compatible(inputs.back(), output)))
        {
            MIGRAPHX_THROW("SELECT_MODULE_INDEX: data inputs must not be tuples; unpack them "
                           "with get_tuple_elem. Only the output buffer may be a tuple.");
        }

        auto mismatched = std::find_if(mods.begin(), mods.end(), [&](module_ref mod) {
            auto names        = get_input_parameter_names(mod);
            auto param_shapes = mod->get_parameter_shapes();
            return not std::equal(
                names.begin(),
                names.end(),
                data_shapes.begin(),
                data_shapes.end(),
                [&](const auto& name, const auto& s) { return param_shapes.at(name) == s; });
        });
        if(mismatched != mods.end())
        {
            MIGRAPHX_THROW("SELECT_MODULE_INDEX: data inputs {" + to_string_range(data_shapes) +
                           "} do not match the parameters of submodule " + (*mismatched)->name() +
                           ".");
        }

        return output;
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
        std::sort(ret.begin(), ret.end());
        return ret;
    }

    static std::size_t find_submodule_index(std::size_t index,
                                            const std::vector<std::size_t>& map,
                                            std::size_t submodule_count)
    {
        if(map.empty())
        {
            if(index >= submodule_count)
            {
                MIGRAPHX_THROW("SELECT_MODULE_INDEX: index " + std::to_string(index) +
                               " is out of range for " + std::to_string(submodule_count) +
                               " submodules.");
            }
            return index;
        }

        auto it = std::find(map.begin(), map.end(), index);
        if(it == map.end())
        {
            MIGRAPHX_THROW("SELECT_MODULE_INDEX: no submodule for index " + std::to_string(index) +
                           ".");
        }
        return std::distance(map.begin(), it);
    }

    argument compute(const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& submodule_list,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        auto index = args.front().at<std::int64_t>();
        if(index < 0)
        {
            MIGRAPHX_THROW("SELECT_MODULE_INDEX: index must be non-negative.");
        }
        const auto idx           = find_submodule_index(index, index_map, submodule_list.size());
        module_ref module_to_run = submodule_list[idx];

        std::unordered_map<std::string, argument> p_map;

        auto in_param_names = get_input_parameter_names(module_to_run);
        // args[0] is the index, a trailing output buffer is only present once
        // lowering appends one, so do not require it here
        assert(in_param_names.size() + 1 <= args.size());
        std::transform(in_param_names.begin(),
                       in_param_names.end(),
                       args.begin() + 1,
                       std::inserter(p_map, p_map.end()),
                       [&](auto&& name, auto&& a) { return std::make_pair(name, a); });

        auto out_param_names    = get_output_parameter_names(module_to_run);
        auto param_shapes       = module_to_run->get_parameter_shapes();
        auto output_sub_objects = args.back().get_sub_objects();
        assert(out_param_names.size() == output_sub_objects.size());
        std::transform(out_param_names.begin(),
                       out_param_names.end(),
                       output_sub_objects.begin(),
                       std::inserter(p_map, p_map.end()),
                       [&](auto&& name, auto&& a) {
                           const auto& ps = param_shapes.at(name);
                           if(a.get_shape() != ps)
                           {
                               MIGRAPHX_THROW("SELECT_MODULE_INDEX: output buffer " +
                                              to_string(a.get_shape()) +
                                              " does not match output parameter " + name + " " +
                                              to_string(ps) + ".");
                           }
                           return std::make_pair(name, a);
                       });
        auto results = run(module_to_run, p_map);
        return argument{results};
    }

    std::vector<std::size_t> output_alias(const std::vector<shape>& shapes) const
    {
        if(not has_output_buffer(shapes))
            return {};
        return {shapes.size() - 1};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
