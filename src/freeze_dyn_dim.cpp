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

#include <migraphx/freeze_dyn_dim.hpp>
#include <migraphx/env.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/errors.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_DYN_DIM_FREEZE_TO)

void freeze_dyn_dim::apply(module_pass_manager& mpm) const
{
    // Read the freeze target N from the env var.  Use the non-cached
    // string form so tests (and runtime) can flip the value between
    // compile passes within a single process without hitting a static
    // cache.  N=0 disables the pass entirely (the default).
    // cppcheck-suppress migraphx-UseCachedEnvVar
    const std::size_t n = value_of("MIGRAPHX_DYN_DIM_FREEZE_TO");
    if(n == 0)
        return;

    module_ref mm    = &mpm.get_module();
    auto param_names = mm->get_parameter_names();

    bool any_replaced = false;
    for(const auto& name : param_names)
    {
        auto old_shape = mm->get_parameter_shape(name);
        if(not old_shape.dynamic())
            continue;

        // Validate N falls inside every non-fixed dyn_dim's [min, max]
        // interval on this parameter.  Throwing here is strictly more
        // useful than silently producing a static shape that the user
        // will never feed at runtime.
        for(const auto& dd : old_shape.dyn_dims())
        {
            if(dd.is_fixed())
                continue;
            auto iv = dd.get_interval();
            if(n < iv.min or n > iv.max)
            {
                MIGRAPHX_THROW("FREEZE_DYN_DIM: MIGRAPHX_DYN_DIM_FREEZE_TO=" + std::to_string(n) +
                               " is outside parameter `" + name + "`'s dynamic-dimension range [" +
                               std::to_string(iv.min) + ", " + std::to_string(iv.max) + "]");
            }
        }

        // shape::to_static replaces every non-fixed dynamic_dimension with
        // a static dimension of size n; fixed dyn_dims (where min==max)
        // are kept as-is.
        auto static_shape = old_shape.to_static(n);

        // Rewrite the parameter in two stages so we never have two live
        // parameters with the same name (which would trip
        // module::insert_parameter's assert in non-NDEBUG builds such
        // as the CI codecov and debug configurations):
        //   1. add a uniquely-named static-shape "stand-in" parameter
        //      and rewire every existing use of the old parameter to it;
        //   2. remove the old parameter -- the original name is now
        //      free -- then add the real static-shape parameter with
        //      the original name and rewire the stand-in to it.
        // The end state is a single static-shape parameter with the
        // original name and the same dataflow graph downstream.
        auto old_param          = mm->get_parameter(name);
        const std::string stand = "__migraphx_freeze_tmp__" + name;
        auto stand_param        = mm->add_parameter(stand, static_shape);
        mm->replace_instruction(old_param, stand_param);
        mm->remove_instruction(old_param);
        auto new_param = mm->add_parameter(name, static_shape);
        mm->replace_instruction(stand_param, new_param);
        mm->remove_instruction(stand_param);
        any_replaced = true;
    }
    (void)any_replaced;
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
