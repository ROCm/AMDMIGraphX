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

#include <migraphx/split_single_dyn_dim.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/matcher.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct dynamic_dimensions_check
{
    std::string dyn_param_str;
    size_t dyn_index;
    size_t min_dim;
    size_t max_dim;
};

optional<dynamic_dimensions_check>
has_one_dyn_dim(const std::unordered_map<std::string, shape>& param_shapes)
{
    // True if parameters contain exactly one dynamic shape with exactly one non-fixed
    // dynamic_dimension.
    auto is_dynamic = [](const auto& p) { return p.second.dynamic(); };
    auto ps_it      = std::find_if(param_shapes.begin(), param_shapes.end(), is_dynamic);
    if(ps_it == param_shapes.end())
        return std::nullopt;
    // Check if there is a second dynamic parameter
    if(std::any_of(std::next(ps_it), param_shapes.end(), is_dynamic))
        return std::nullopt;
    const auto& dds = ps_it->second.dyn_dims();

    auto is_non_fixed = [](const auto& dd) { return not dd.is_fixed(); };
    auto dds_it       = std::find_if(dds.begin(), dds.end(), is_non_fixed);
    if(dds_it == dds.end())
        return std::nullopt;
    // Check if there is a second non-fixed dynamic_dimension
    if(std::any_of(std::next(dds_it), dds.end(), is_non_fixed))
        return std::nullopt;
    return dynamic_dimensions_check{ps_it->first,
                                    static_cast<std::size_t>(std::distance(dds.begin(), dds_it)),
                                    dds_it->min,
                                    dds_it->max};
}

namespace {
struct find_static_2in_broadcasts
{
    // Convert 2 input static shape broadcast/multibroadcast into 1 input version.
    // Some compiler passes (ex. simplify_algebra) only support the 1 input versions
    // of the broadcasting operators.
    auto matcher() const
    {
        return match::broadcast(match::nargs(2),
                                match::arg(0)(match::static_shape()),
                                match::arg(1)(match::static_shape()));
    }

    void apply(module& m, const match::matcher_result& mr) const
    {
        auto ins          = mr.result;
        auto out_lens     = ins->get_shape().lens();
        auto broadcast_op = ins->get_operator();
        if(broadcast_op.name() == "broadcast")
        {
            broadcast_op.from_value({{"out_lens", out_lens}});
        }
        else
        {
            broadcast_op.from_value({{"out_lens", out_lens}, {"out_dyn_dims", {}}});
        }
        m.replace_instruction(ins, broadcast_op, ins->inputs().at(0));
    }
};
} // namespace

/**
 * Makes all the shapes in the dynamic_dimension range.  Probably won't work for `if`
 * and `loop` instructions, depending on how the submodules for those
 * work. Inserts select_module instruction to the top. Replaces return, bypassing other
 * instructions. Skips if the dynamic parameter outputs to a select_module operator.
 */
void split_single_dyn_dim::apply(module_pass_manager& mpm) const
{
    module_ref mm                               = &mpm.get_module();
    auto param_names                            = mm->get_parameter_names();
    auto param_shapes                           = mm->get_parameter_shapes();
    optional<dynamic_dimensions_check> dd_check = has_one_dyn_dim(param_shapes);
    auto any_sm_next                            = [&](auto ddc) {
        auto p_outputs = mm->get_parameter(ddc->dyn_param_str)->outputs();
        return std::any_of(p_outputs.cbegin(), p_outputs.cend(), [](auto ins) {
            return ins->name() == "select_module";
        });
    };
    if(dd_check.has_value() and not any_sm_next(dd_check))
    {
        const auto& dyn_param = mm->get_parameter(dd_check->dyn_param_str);
        auto dyn_param_shape  = mm->get_parameter_shape(dd_check->dyn_param_str);
        std::vector<module_ref> submodules;
        // create submodules for each dimension size
        for(size_t dim_size : migraphx::range(dd_check->min_dim, dd_check->max_dim + 1))
        {
            auto* submod = mpm.create_module("dim_" + std::to_string(dim_size));
            // instruction map for new static shaped submodule parameters
            std::unordered_map<instruction_ref, instruction_ref> map_ins;
            // create static shape using dim_size
            auto static_lens                    = dyn_param_shape.max_lens();
            static_lens.at(dd_check->dyn_index) = dim_size;
            map_ins[dyn_param]                  = submod->add_parameter(
                dd_check->dyn_param_str, migraphx::shape{dyn_param_shape.type(), static_lens});
            auto outputs = submod->add_instructions(mm, map_ins);
            submod->add_return({outputs});
            match::find_matches(*submod, find_static_2in_broadcasts{});
            submodules.push_back(submod);
        }
        // redirect to select_module operator and return
        std::vector<instruction_ref> sm_inputs;
        std::transform(param_names.cbegin(),
                       param_names.cend(),
                       std::back_inserter(sm_inputs),
                       [&](auto pn) { return mm->get_parameter(pn); });
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
