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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

/// Runs one submodule chosen by a host integer.
///
/// select_module_index(index, data..., output) [submodules...]
/// An empty index_map uses the integer as the submodule position. Otherwise
/// submodule i runs when the integer equals index_map[i].
struct select_module_index
{
    std::vector<std::size_t> index_map;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.index_map, "index_map"));
    }

    std::string name() const { return "select_module_index"; }

    shape compute_shape(const std::vector<shape>& inputs,
                        const std::vector<module_ref>& mods) const
    {
        check_shapes{inputs, *this}.has_at_least(1);
        const auto& index = inputs.front();
        auto index_type   = index.type();
        // On GPU the index is a hip::load_scalar result; under ref it is a host integer.
        if(index.elements() != 1 or
           (index_type != shape::int32_type and index_type != shape::int64_type))
        {
            MIGRAPHX_THROW("SELECT_MODULE_INDEX: index must be a static single-element "
                           "int32 or int64.");
        }
        if(mods.empty())
        {
            MIGRAPHX_THROW("SELECT_MODULE_INDEX: operator requires at least one submodule.");
        }
        if(not index_map.empty() and index_map.size() != mods.size())
        {
            MIGRAPHX_THROW("SELECT_MODULE_INDEX: index_map must match submodule count.");
        }

        auto out_shapes0 = mods.front()->get_output_shapes();
        for(std::size_t i = 1; i < mods.size(); ++i)
        {
            auto out_shapes = mods[i]->get_output_shapes();
            if(not std::equal(
                   out_shapes.begin(), out_shapes.end(), out_shapes0.begin(), out_shapes0.end()))
            {
                MIGRAPHX_THROW(
                    "SELECT_MODULE_INDEX: output shapes of submodules must be the same.");
            }
        }

        return shape{out_shapes0};
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
            MIGRAPHX_THROW("SELECT_MODULE_INDEX: no submodule for index " +
                           std::to_string(index) + ".");
        }
        return std::distance(map.begin(), it);
    }

    argument compute(const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& submodule_list,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        module_ref module_to_run{};
        args.front().visit_at([&](auto index) {
            if(index < 0)
            {
                MIGRAPHX_THROW("SELECT_MODULE_INDEX: index must be non-negative.");
            }
            const auto idx = find_submodule_index(index, index_map, submodule_list.size());
            module_to_run = submodule_list[idx];
        });

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
                           auto ps = param_shapes.at(name);
                           if(a.get_shape() != ps)
                           {
                               assert(ps.bytes() <= a.get_shape().bytes());
                               return std::make_pair(name, a.reshape(ps));
                           }
                           return std::make_pair(name, a);
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
