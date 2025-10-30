/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/split_single_dyn_dim.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/matcher.hpp>
#include <migraphx/bit.hpp>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_USE_FIXED_PAD);


void rewrite_pure_dyn_params(module& m)
{
    std::vector<instruction_ref> remove_parameters;
    auto params                           = m.get_parameters();
    for(auto param : params){
        auto shape = param->get_shape();
        if(not shape.dynamic())
            continue;
        if(not std::all_of(shape.dyn_dims().begin(), shape.dyn_dims().end(), [](auto dd) {
            return dd.min == 1 and dd.max == std::numeric_limits<std::size_t>::max();
        }))
            continue;
        if(not std::all_of(param->outputs().begin(), param->outputs().end(), [&](auto output){
            return output->get_shape() == param->outputs().front()->get_shape();
        }))
            MIGRAPHX_THROW("unsupported multi output shapes for purely dynamic shapes");
        auto out_shape = param->outputs().front()->get_shape();
        std::string param_name = param->get_operator().to_value()["parameter"].to<std::string>();
        m.rename_parameter(param, param_name + "_old");
        auto new_param = m.add_parameter(param_name, out_shape);
        m.replace_instruction(param, new_param);
        remove_parameters.push_back(param);
    }

    for(const auto& i : remove_parameters)
        m.remove_instruction(i);

}

struct dynamic_dimensions_check
{
    std::string dyn_param_str;
    shape::dynamic_dimension dd;
};

/**
 * Returns value if the parameters contain non-fixed dynamic_dimensions that are the same between
 * all of the dynamic shape parameters.
 * In other words, each parameter can have one non-fixed dynamic_dimension `x` where `x` is the same
 * between all of the parameters with a non-fixed dynamic_dimension.
 * Returns the parameters and the dynamic dimension in a vector of dynamic_dimensions_check objects.
 */
static optional<std::vector<dynamic_dimensions_check>>
has_one_unique_dyn_dim(const std::unordered_map<std::string, shape>& param_shapes)
{
    auto is_dynamic = [](const auto& p) { return p.second.dynamic(); };
    std::vector<std::decay_t<decltype(param_shapes)>::value_type> dyn_params{};
    std::copy_if(
        param_shapes.begin(), param_shapes.end(), std::back_inserter(dyn_params), is_dynamic);
    if(dyn_params.empty())
        return std::nullopt;
    std::vector<dynamic_dimensions_check> ret{};
    // get non-fixed dynamic_dimension from all parameters
    for(const auto& param : dyn_params)
    {
        const auto& dds    = param.second.dyn_dims();
        auto num_non_fixed = std::count_if(dds.cbegin(), dds.cend(), [&](const auto& dd) {
            if(not dd.is_fixed())
            {
                ret.push_back(dynamic_dimensions_check{param.first, dd});
                return true;
            }
            return false;
        });
        // catch more than one non-fixed dynamic_dimension
        if(num_non_fixed > 1)
        {
            return std::nullopt;
        }
    }
    if(ret.empty())
    {
        return std::nullopt;
    }
    // check all the same dynamic_dimension
    bool same_dd = std::all_of(
        ret.begin() + 1, ret.end(), [&](const auto& ddc) { return ddc.dd.min == ret.at(0).dd.min and 
                                                            ddc.dd.max == ret.at(0).dd.max; });    
    if(same_dd)
    {
        return ret;
    }
    return std::nullopt;
}

/**
 * Check the parameters in std::vector<dynamic_dimensions_check> object to see if any of the
 * parameters outputs to a select_module operator.
 */
static bool any_sm_next(const_module_ref mm, const std::vector<dynamic_dimensions_check>& ddcs)
{
    if(any_of(mm->begin(), mm->end(), [](auto ins) { return ins.name() == "select_module";} ))
    {
        return true;
    }
    for(const auto& ddc : ddcs)
    {
        auto p_outputs  = mm->get_parameter(ddc.dyn_param_str)->outputs();
        auto has_fixed_pad = [&](const std::vector<instruction_ref>& outputs){
            return std::any_of(outputs.cbegin(), outputs.cend(), [](auto ins) {
            // TODO: need to traverse parameter outputs properly and look for
            //  any possible signs of this run already being done...
            // or better if there was a module attribute that says it's already done or not
            return ins->name() == "fixed_pad";
            });
        };
        bool is_sm_next = std::any_of(p_outputs.cbegin(), p_outputs.cend(), [&](auto ins) {
            if(ins->name() == "convert")
                return has_fixed_pad(ins->outputs());
            return ins->name() == "fixed_pad";
        });
        if(is_sm_next)
        {
            return true;
        };
    }
    return false;
}
/*
 * Returns a vector of all powers of 2 between min and max.
 * Currently disabled for now.
 */
// static std::vector<size_t> powers_of_2_between(size_t min, size_t max)
// {
//     std::vector<size_t> result;
//     for(size_t p = bit_ceil(min + 1); p < max; p *= 2)
//     {
//         if(p > min)
//         {
//             result.push_back(p);
//         }
//     }
//     return result;
// }

static void insert_fixed_pad(module& m, const migraphx::shape::dynamic_dimension& dyn_dim)
{
    size_t max_pad_val = dyn_dim.max;
    for(auto&& param : m.get_parameters())
    {
        auto param_shape = param->get_shape();
        if(not param_shape.dynamic())
            continue;
        auto static_shape = param_shape.to_static(max_pad_val);
        auto fixed_pad = m.insert_instruction(std::next(param),
            make_op("fixed_pad", {{"output_lens", static_shape.lens()}}), param);
        m.replace_instruction(param, fixed_pad);
    }
}

/**
 * Makes all the shapes in the dynamic_dimension range.  Probably won't work for `if`
 * and `loop` instructions, depending on how the submodules for those
 * work. Inserts select_module instruction to the top. Replaces return, bypassing other
 * instructions. Skips if the dynamic parameter outputs to a select_module operator.
 */
void split_single_dyn_dim::apply(module_pass_manager& mpm) const
{
    module_ref mm                               = &mpm.get_module();
    rewrite_pure_dyn_params(*mm);
    auto param_names                            = mm->get_parameter_names();
    auto param_shapes                           = mm->get_parameter_shapes();
    optional<std::vector<dynamic_dimensions_check>> dd_check_vec =
        has_one_unique_dyn_dim(param_shapes);
    if(dd_check_vec.has_value() and not any_sm_next(mm, dd_check_vec.value()))
    {
        // all dynamic dimension objects should be the same for all parameters in dd_check_vec
        auto dyn_dim = dd_check_vec->at(0).dd;

        if(enabled(MIGRAPHX_USE_FIXED_PAD{}))
        {
            insert_fixed_pad(*mm, dyn_dim);
            return;
        }
        // create submodules for the range of dimension sizes
        // use min, max, (and powers of 2 in between currently disabled),
        // and any user-supplied optimals
        std::vector<module_ref> submodules;
        std::set<size_t> dim_sizes{dyn_dim.min, dyn_dim.max};
        // std::vector<size_t> powers_of_2 = powers_of_2_between(dyn_dim.min, dyn_dim.max);
        // dim_sizes.insert(powers_of_2.begin(), powers_of_2.end());
        if(dyn_dim.has_optimal())
        {
            dim_sizes.insert(dyn_dim.optimals.begin(), dyn_dim.optimals.end());
        }
        size_t prev_dim_size = 0;
        for(size_t dim_size : dim_sizes)
        {
            auto* submod = mpm.create_module("dim_" + std::to_string(dim_size));
            // instruction map for new static shaped submodule parameters
            std::unordered_map<instruction_ref, instruction_ref> map_ins;
            for(const auto& dd_check : dd_check_vec.value())
            {
                // create static shape using dim_size
                const auto& dyn_param = mm->get_parameter(dd_check.dyn_param_str);
                auto dyn_param_shape  = mm->get_parameter_shape(dd_check.dyn_param_str);
                auto new_dyn_dims = dyn_param_shape.dyn_dims();
                for(auto& new_dd : new_dyn_dims)
                {
                    if(not new_dd.is_fixed())
                    {
                        new_dd.min = prev_dim_size + 1;
                        new_dd.max = dim_size;
                        new_dd.optimals = {};
                    }
                }
                auto new_dyn_shape = shape{dyn_param_shape.type(), new_dyn_dims };
                auto static_shape     = dyn_param_shape.to_static(dim_size);
                auto new_dyn_param = submod->add_parameter(dd_check.dyn_param_str, new_dyn_shape);
                map_ins[dyn_param]    = submod->add_instruction(
                        make_op("fixed_pad", {{"output_lens", static_shape.lens()}}), new_dyn_param);
                
            }
            // insert static parameters

            for(auto&& param_name : param_names)
            {
                // TODO would there ever be a tuple input param?
                if(not mm->get_parameter_shape(param_name).any_of_dynamic())
                {
                    auto static_param = mm->get_parameter(param_name);
                    map_ins[static_param] = submod->add_parameter(param_name, static_param->get_shape());
                }
            }
            auto outputs = submod->add_instructions(mm, &map_ins);
            submod->add_return({outputs});
            submodules.push_back(submod);
            prev_dim_size = dim_size;
        }
        // sort parameters by name for consistency (vs. parameter order attr)
        std::sort(param_names.begin(), param_names.end());
        // redirect to select_module operator and return
        std::vector<instruction_ref> sm_inputs;
        std::transform(param_names.cbegin(),
                       param_names.cend(),
                       std::back_inserter(sm_inputs),
                       [&](auto pn) { return mm->get_parameter(std::move(pn)); });
        auto output_shapes       = mm->get_output_shapes();
        migraphx::shape out_attr = migraphx::shape{output_shapes};
        auto sm_ins              = mm->add_instruction(
            migraphx::make_op("select_module",
                              {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
            sm_inputs,
            submodules);
        std::vector<instruction_ref> outputs(output_shapes.size());
        for(size_t i = 0; i < output_shapes.size(); ++i)
        {
            outputs.at(i) =
                mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", i}}), sm_ins);
        }
        mm->replace_return(outputs);
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
