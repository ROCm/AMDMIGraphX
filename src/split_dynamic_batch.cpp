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

#include <migraphx/split_dynamic_batch.hpp>
#include <migraphx/functional.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

bool has_one_dyn_dim(std::unordered_map<std::string, shape> param_shapes,
                     std::string& dyn_param_str,
                     unsigned int& max_batches)
{
    // true if exactly one dynamic shape with exactly one non-fixed dynamic_dimension
    // dyn_param_str is updated to the parameter string with the dynamic_dimension
    if(std::none_of(
           param_shapes.cbegin(), param_shapes.cend(), [](auto ps) { return ps.second.dynamic(); }))
        return false;
    int num_dynamic = 0;
    std::string out_str;
    int tmp_batches;
    for(auto ps : param_shapes)
    {
        if(ps.second.dynamic())
        {
            num_dynamic += 1;
            if(num_dynamic > 1)
            {
                return false;
            }
            int num_nf = 0;
            for(auto dd : ps.second.dyn_dims())
            {
                if(not dd.is_fixed())
                {
                    num_nf += 1;
                    tmp_batches = dd.max;
                }
            }
            if(num_nf == 1)
            {
                out_str = ps.first;
            }
            else
            {
                return false;
            }
        }
    }
    dyn_param_str = out_str;
    max_batches   = tmp_batches;
    return true;
}

// don't use program
// use module_pass_manager
void split_dynamic_batch::apply(program& p) const
{
    auto param_shapes = p.get_parameter_shapes();
    std::string dyn_param_str;
    unsigned int max_batches;
    if(has_one_dyn_dim(param_shapes, dyn_param_str, max_batches))
    {
        // floor(log_2(max_batches))
        unsigned int mask = 1U << 31;
        while(not(max_batches & mask))
        {
            mask >>= 1;
        }
        // create submodules based on binary base
        while(mask)
        {
            auto mod =
                p.copy_module(p.get_main_module()->name(), "batch_size_" + std::to_string(mask));
            // change dynamic input parameter shape to static
            // propagate the shape change through the model
            // create new parameter_list
            // use module.add_instructions() with map between old and new static parameter
            // add a return to the submodule
            mask >>= 1;
        }
        // create additional submodules for optimal values if not already done
        //
        // delete stuff in main module, insert instruction to the top, replace return bypassing
        // other instructions unused instructions should be removed by dead_code_elimination
    }
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
