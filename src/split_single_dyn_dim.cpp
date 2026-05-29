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

#include <migraphx/split_single_dyn_dim.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/matcher.hpp>
#include <set>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

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
        ret.begin() + 1, ret.end(), [&](const auto& ddc) { return ddc.dd == ret.at(0).dd; });
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
    for(const auto& ddc : ddcs)
    {
        auto p_outputs  = mm->get_parameter(ddc.dyn_param_str)->outputs();
        bool is_sm_next = std::any_of(p_outputs.cbegin(), p_outputs.cend(), [](auto ins) {
            return ins->name() == "select_module";
        });
        if(is_sm_next)
        {
            return true;
        };
    }
    return false;
}

/**
 * Makes all the shapes in the dynamic_dimension range.  Probably won't work for `if`
 * and `loop` instructions, depending on how the submodules for those
 * work. Inserts select_module instruction to the top. Replaces return, bypassing other
 * instructions. Skips if the dynamic parameter outputs to a select_module operator.
 */
void split_single_dyn_dim::apply(module_pass_manager& mpm) const
{
    module_ref mm     = &mpm.get_module();
    auto param_names  = mm->get_parameter_names();
    auto param_shapes = mm->get_parameter_shapes();
    optional<std::vector<dynamic_dimensions_check>> dd_check_vec =
        has_one_unique_dyn_dim(param_shapes);
    if(dd_check_vec.has_value() and not any_sm_next(mm, dd_check_vec.value()))
    {
        // all dynamic dimension objects should be the same for all parameters in dd_check_vec
        auto dyn_dim = dd_check_vec->at(0).dd;
        // Only emit submodules at the interval endpoints and any user-supplied
        // optimals, instead of one submodule per integer in [min, max]. Cuts
        // compile cost dramatically on wide ranges while preserving the most
        // commonly-hit batch sizes.
        std::vector<module_ref> submodules;
        auto dim_interval = dyn_dim.get_interval();
        std::set<std::size_t> dim_sizes{dim_interval.min, dim_interval.max};
        if(dyn_dim.has_optimal())
        {
            auto opts = dyn_dim.get_optimals();
            dim_sizes.insert(opts.begin(), opts.end());
        }
        // Each bucket submodule "dim_<N>" accepts inputs whose non-fixed
        // dimension lies in (prev_dim_size, dim_size] and pads them up to a
        // static shape of length dim_size via fixed_pad. select_module routes
        // inputs to the smallest bucket whose dim_size >= orig_batch, so each
        // submodule only ever sees inputs in its own range.
        std::size_t prev_dim_size = 0;
        for(std::size_t dim_size : dim_sizes)
        {
            auto* submod = mpm.create_module("dim_" + std::to_string(dim_size));
            std::unordered_map<instruction_ref, instruction_ref> map_ins;
            for(const auto& dd_check : dd_check_vec.value())
            {
                const auto& dyn_param = mm->get_parameter(dd_check.dyn_param_str);
                auto dyn_param_shape  = mm->get_parameter_shape(dd_check.dyn_param_str);
                // Narrow the non-fixed dim to this bucket's range; keep any
                // already-fixed dims untouched.
                auto new_dyn_dims = dyn_param_shape.dyn_dims();
                for(auto& new_dd : new_dyn_dims)
                {
                    if(not new_dd.is_fixed())
                        new_dd = shape::dynamic_dimension{prev_dim_size + 1, dim_size};
                }
                auto new_dyn_shape =
                    shape{dyn_param_shape.type(), std::move(new_dyn_dims)};
                auto new_dyn_param =
                    submod->add_parameter(dd_check.dyn_param_str, new_dyn_shape);
                // fixed_pad expands the dynamic input to its max_lens (==
                // dim_size for the non-fixed axis), giving the rest of the
                // submodule a static shape to operate on.
                map_ins[dyn_param] =
                    submod->add_instruction(make_op("fixed_pad"), new_dyn_param);
            }
            // Preserve names of static (non-dynamic) parameters so
            // select_module can map them positionally by name.
            for(const auto& param_name : param_names)
            {
                if(not mm->get_parameter_shape(param_name).any_of_dynamic())
                {
                    auto static_param = mm->get_parameter(param_name);
                    map_ins[static_param] =
                        submod->add_parameter(param_name, static_param->get_shape());
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
