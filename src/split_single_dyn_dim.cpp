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

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool has_one_dyn_dim(const std::unordered_map<std::string, shape>& param_shapes,
                     std::string& dyn_param_str,
                     size_t& dyn_index,
                     size_t& min_dim,
                     size_t& max_dim)
{
    // True if parameters contain exactly one dynamic shape with exactly one non-fixed
    // dynamic_dimension.
    size_t num_dynamic = 0;
    for(const auto& ps : param_shapes)
    {
        if(ps.second.dynamic())
        {
            num_dynamic += 1;
            if(num_dynamic > 1)
            {
                return false;
            }
            size_t num_nf = 0;
            auto dds      = ps.second.dyn_dims();
            for(size_t i = 0; i < dds.size(); ++i)
            {
                const auto& dd = dds.at(i);
                if(not dd.is_fixed())
                {
                    num_nf += 1;
                    min_dim   = dd.min;
                    max_dim   = dd.max;
                    dyn_index = i;
                }
            }
            if(num_nf == 1)
            {
                dyn_param_str = ps.first;
            }
            else
            {
                return false;
            }
        }
    }
    return (num_dynamic == 1);
}

/**
 * Makes all the shapes in the dynamic_dimension range.
 * Probably won't work for `if` and `loop` instructions, depending on how the submodules for those
 * work. Inserts select_module instruction to the top, replace return bypassing other instructions.
 */
void split_single_dyn_dim::apply(module_pass_manager& mpm) const
{
    module_ref mm     = &mpm.get_module();
    auto param_names  = mm->get_parameter_names();
    auto param_shapes = mm->get_parameter_shapes();
    std::string dyn_param_name;
    size_t dyn_index;
    size_t min_dim;
    size_t max_dim;
    if(has_one_dyn_dim(param_shapes, dyn_param_name, dyn_index, min_dim, max_dim))
    {
        const auto& dyn_param = mm->get_parameter(dyn_param_name);
        auto dyn_param_shape  = mm->get_parameter_shape(dyn_param_name);
        std::vector<module_ref> submodules;
        // create submodules for each dimension size
        for(size_t dim_size = min_dim; dim_size <= max_dim; ++dim_size)
        {
            auto* submod = mpm.create_module("dim_" + std::to_string(dim_size));
            // instruction map for new static shaped submodule parameters
            std::unordered_map<instruction_ref, instruction_ref> map_ins;
            // create static shape using dim_size
            auto static_lens          = dyn_param_shape.max_lens();
            static_lens.at(dyn_index) = dim_size;
            auto static_param         = submod->add_parameter(
                dyn_param_name, migraphx::shape{dyn_param_shape.type(), static_lens});
            map_ins[dyn_param] = static_param;
            auto outputs       = submod->add_instructions(mm, map_ins);
            submod->add_return({outputs});
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
