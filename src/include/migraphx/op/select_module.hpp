/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

    shape compute_shape(const std::vector<shape>&, const std::vector<module_ref>&) const
    {
        return shape{output_dyn_shapes};
    }

    argument compute(const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& submodule_list,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        // find submodule with input parameter shapes exactly the same as the input arguments
        // assuming arguments are in the same order as the input parameters
        auto module_iter =
            std::find_if(submodule_list.cbegin(), submodule_list.cend(), [&](module_ref mr) {
                auto param_names = mr->get_parameter_names();
                assert(param_names.size() <= args.size());
                return std::equal(param_names.cbegin(),
                                  param_names.cend(),
                                  args.cbegin(),
                                  [&](auto p_name, auto a) {
                                      return a.get_shape() == mr->get_parameter_shape(p_name);
                                  });
            });

        if(module_iter == submodule_list.end())
        {
            MIGRAPHX_THROW("SELECT_MODULE: no compatible submodules found for given input shapes");
        }
        auto* module_to_run = *module_iter;
        std::unordered_map<std::string, argument> params;

        // add input parameters
        auto param_names = module_to_run->get_parameter_names();
        assert(param_names.size() <= args.size());
        std::transform(param_names.begin(),
                       param_names.end(),
                       args.begin(),
                       std::inserter(params, params.end()),
                       [](auto&& name, auto&& a) { return std::make_pair(name, a); });

        auto results = run(module_to_run, params);
        return argument{results};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
