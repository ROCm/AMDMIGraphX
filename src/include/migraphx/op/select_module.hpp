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
#include <migraphx/dyn_output.hpp>
#include <set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {


// Make this work just for exact matches
// can get rid of the other attributes and just check all the parameters are the same
// GPU version of this might have to deal with output parameters
// see loop op for how the output parameters are dealt with there
// Can have multiple inputs but only one output?
struct select_module
{
    // output shape of the dynamic model
    shape output_dyn_shape;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_dyn_shape, "output_dyn_shape"));
    }

    std::string name() const { return "select_module"; }

    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true};
		return shape{output_dyn_shape};
    }

    argument compute(const shape&,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& submodule_list,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
		// find submodule with parameter shapes exactly the same as the input arguments
		// assuming arguments are in the same order as the parameters
        auto module_to_run = std::find_if(submodule_list.begin(), submodule_list.end(), [&](module_ref mr) {
			auto param_names = mr.get_parameter_names();
			std::equal(args.cbegin(), args.cend(), param_names.cbegin(), [&](auto a, auto p_name) {
				return a.get_shape() == mr.get_parameter_shape(p_name);
			});
		});

        if(module_to_run == submodule_list.end())
        {
            MIGRAPHX_THROW("SELECT_MODULE: no compatible submodules found for given input shapes");
        }
			
        auto param_names = module_to_run.get_parameter_names();
        assert(pnames.size() <= args.size());
        std::unordered_map<std::string, argument> params;
        std::transform(param_names.begin(),
                       param_names.end(),
                       args.begin(),
                       std::inserter(params, params.end()),
                       [](auto&& name, auto&& arg) { return std::make_pair(name, arg); });

        auto results = run(module_to_run, params);
        return argument{results};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
