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
#include <migraphx/optional.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct select_module
{
    shape output_dyn_shapes;
    // Optional names for the input arguments. When set, a submodule binds each of its
    // input parameters to the argument at that name's position here, so submodules may
    // declare different parameter subsets. When empty, arguments bind positionally to
    // each submodule's sorted parameter names.
    std::vector<std::string> param_names{};

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_dyn_shapes, "output_dyn_shapes"),
                    f(self.param_names, "param_names"));
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

    // The argument holding the value of a named submodule parameter: the argument at the
    // name's position in param_names, or by sorted-name position when param_names is
    // empty. Returns nullopt for a name a submodule declares but no argument drives.
    optional<argument>
    find_arg(const std::string& name, std::size_t position, const std::vector<argument>& args) const
    {
        if(param_names.empty())
        {
            assert(position < args.size());
            return args[position];
        }
        auto it = std::find(param_names.begin(), param_names.end(), name);
        if(it == param_names.end())
            return nullopt;
        auto index = std::distance(param_names.begin(), it);
        assert(index < args.size());
        return args[index];
    }

    argument compute(const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& submodule_list,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        // Find the first submodule whose input parameter shapes exactly match the
        // arguments that drive them.
        auto module_iter =
            std::find_if(submodule_list.cbegin(), submodule_list.cend(), [&](module_ref mr) {
                auto in_param_names = get_input_parameter_names(mr);
                auto param_shapes   = mr->get_parameter_shapes();
                assert(in_param_names.size() <= args.size());
                std::size_t position = 0;
                return std::all_of(
                    in_param_names.cbegin(), in_param_names.cend(), [&](const auto& p_name) {
                        auto arg = find_arg(p_name, position++, args);
                        return arg.has_value() and arg->get_shape() == param_shapes[p_name];
                    });
            });

        if(module_iter == submodule_list.end())
        {
            MIGRAPHX_THROW("SELECT_MODULE: no compatible submodules found for given input shapes");
        }

        auto* module_to_run = *module_iter;
        std::unordered_map<std::string, argument> p_map;

        // add input parameters to parameter_map
        auto in_param_names = get_input_parameter_names(module_to_run);
        assert(in_param_names.size() <= args.size());
        std::size_t position = 0;
        std::transform(in_param_names.begin(),
                       in_param_names.end(),
                       std::inserter(p_map, p_map.end()),
                       [&](auto&& name) {
                           auto arg = find_arg(name, position++, args);
                           assert(arg.has_value());
                           return std::make_pair(name, *arg);
                       });

        // One tuple output parameter in main module to multiple output parameters in submodule
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
                           else
                           {
                               return std::make_pair(name, a);
                           }
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
