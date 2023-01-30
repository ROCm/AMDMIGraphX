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

struct select_module
{
    // output shape of the dynamic model
    shape output_dyn_shape;
    int input_batch_index  = -1;
    int output_batch_index = -1;
    std::string dyn_batch_param_name;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.output_dyn_shape, "output_dyn_shape"),
                    f(self.input_batch_index, "input_batch_index"),
                    f(self.output_batch_index, "output_batch_index"),
                    f(self.dyn_batch_param_name, "dyn_batch_param_name"));
    }

    std::string name() const { return "select_module"; }

    // this should run once during model compilation with dynamic shape input
    // run once on each model evaluation with static shape input
    shape compute_shape(std::vector<shape> inputs) const
    {
        check_shapes{inputs, *this, true}.has(1);
        auto s0 = inputs.at(0);
        if(s0.dynamic())
        {
            // should we check that the submodules have the same parameters here?
            // check that no more than one parameter is non-fixed?
            // would need to use version of compute_shape with the parameter list
            return shape{output_dyn_shape};
        }
        else
        {
            auto batch_size            = s0.lens().at(input_batch_index);
            auto dds                   = output_dyn_shape.dyn_dims();
            dds.at(output_batch_index) = {batch_size, batch_size};
            std::vector<std::size_t> dims;
            if(std::all_of(dds.begin(), dds.end(), [](auto dd) { return dd.is_fixed(); }))
            {
                std::transform(
                    dds.begin(), dds.end(), std::back_inserter(dims), [](auto d) { return d.max; });
                return {output_dyn_shape.type(), dims};
            }
            else
            {
                MIGRAPHX_THROW("SELECT_MODULE: more than one input dimension was non-fixed");
            }
        }
    }

    argument compute(const dyn_output& dyn_out,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& submodule_list,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        std::vector<module_ref> modules_to_run;
        for(const auto& mod : submodule_list)
        {
            // find submodule with the same parameter shape as the input data
            auto p_shape = mod->get_parameter_shape(dyn_batch_param_name);
            if(p_shape == dyn_out.computed_shape)
            {
                modules_to_run.push_back(mod);
                break;
            }
        }
        // TODO if an exact match is not found, assemble module list from binary base

        if(modules_to_run.empty())
        {
            MIGRAPHX_THROW("SELECT_MODULE: no compatible submodules found");
        }
        std::set<std::string> pnames;
        for(const auto& mod : modules_to_run)
        {
            // If all the modules have the same parameters, this would only need to run once
            auto names = mod->get_parameter_names();
            pnames.insert(names.begin(), names.end());
        }

        assert(pnames.size() < args.size());
        std::unordered_map<std::string, argument> params;
        std::transform(pnames.begin(),
                       pnames.end(),
                       args.begin(),
                       std::inserter(params, params.end()),
                       [](auto&& name, auto&& arg) { return std::make_pair(name, arg); });

        // TODO run multiple modules and split the parameter data to each batch size
        auto results = run(modules_to_run.at(0), params);
        return argument{results};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
