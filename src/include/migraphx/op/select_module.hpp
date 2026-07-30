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
#include <migraphx/module.hpp>
#include <migraphx/ranges.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct select_module
{
    shape output_dyn_shapes;

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

    static bool parameter_shape_matches(const shape& actual, const shape& expected)
    {
        if(not expected.dynamic())
            return actual == expected;
        if(actual.type() != expected.type() or not shape::is_compatible_lens(actual, expected))
            return false;
        if(not expected.symbolic())
            return true;
        if(actual.dynamic())
            return actual == expected;

        std::unordered_map<sym::expr, std::size_t> bindings;
        auto indices = migraphx::range(expected.ndim());
        transform_if(
            indices.begin(),
            indices.end(),
            std::inserter(bindings, bindings.end()),
            [&](auto i) { return expected.dyn_dims()[i].sym_expr.name() != "literal"; },
            [&](auto i) {
                return std::make_pair(sym::as_symbol(expected.dyn_dims()[i].sym_expr),
                                      actual.lens()[i]);
            });
        return shape::is_compatible(actual, expected.to_static(bindings));
    }

    argument compute(const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& submodule_list,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        // Select the first submodule whose sorted input parameters match the argument shapes.
        // Static shapes require equality; dynamic shapes require matching types and compatible
        // bounds.
        auto module_iter =
            std::find_if(submodule_list.cbegin(), submodule_list.cend(), [&](module_ref mr) {
                auto in_param_names = get_input_parameter_names(mr);
                auto param_shapes   = mr->get_parameter_shapes();
                assert(in_param_names.size() <= args.size());
                return std::equal(in_param_names.cbegin(),
                                  in_param_names.cend(),
                                  args.cbegin(),
                                  [&](const auto& p_name, const auto& a) {
                                      const auto& expected = param_shapes.at(p_name);
                                      const auto& actual   = a.get_shape();
                                      return parameter_shape_matches(actual, expected);
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
        std::transform(in_param_names.begin(),
                       in_param_names.end(),
                       args.begin(),
                       std::inserter(p_map, p_map.end()),
                       [&](auto&& name, auto&& a) { return std::make_pair(name, a); });

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
