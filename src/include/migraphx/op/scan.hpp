/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_OPERATORS_LOOP_HPP
#define MIGRAPHX_GUARD_OPERATORS_LOOP_HPP

#include <migraphx/op/name.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/config.hpp>
#include <migraphx/module.hpp>
#include <migraphx/run_loop.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <cmath>
#include <string>
#include <utility>
#include <set>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace op {

struct scan : op_name<scan>
{
    int64_t iterations;
    int64_t num_scan_inputs;
    int64_t num_state_vars;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.iterations, "iterations"),
                    f(self.num_scan_inputs, "num_scan_inputs"),
                    f(self.num_state_vars, "num_state_vars"));
    }

    shape compute_shape(const std::vector<shape>& inputs, std::vector<module_ref> mods) const
    {
        assert(mods.size() == 1);
        check_shapes{inputs, *this}.standard();
        auto mod = mods.front();
        // The module has N + K outputs
        auto mod_output_shapes = mod->get_output_shapes();
        std::vector<shape> op_output_shapes{mod_output_shapes.begin(),
                                            mod_output_shapes.begin() + num_state_vars};
        auto K = mod_output_shapes.size() - num_state_vars;
        op_output_shapes.reserve(num_state_vars + iterations * K);
        for(auto i = 0; i < iterations; ++i)
            op_output_shapes.insert(op_output_shapes.end(),
                                    mod_output_shapes.begin() + num_state_vars,
                                    mod_output_shapes.end());

        return shape{op_output_shapes};
    }

    std::unordered_map<std::string, int> get_output_params(const module_ref mod) const
    {
        std::unordered_map<std::string, int> ret;
        const std::string output_prefix = "#output_";

        const auto& param_names = mod->get_parameter_names();
        for(const auto& name : param_names)
        {
            auto n = name.find(output_prefix);
            if(n == std::string::npos)
                continue;
            int idx   = std::stoi(name.substr(n + output_prefix.size()));
            ret[name] = idx;
        }

        return ret;
    }

    argument compute(context& ctx,
                     const shape& out_shape,
                     const std::vector<argument>& args,
                     const std::vector<module_ref>& mods,
                     const std::function<std::vector<argument>(
                         module_ref&, const std::unordered_map<std::string, argument>&)>& run) const
    {
        assert(mods.size() == 1);
        auto mod          = mods.front();
        auto param_shapes = mod->get_parameter_shapes();
        auto param_names  = mod->get_parameter_names();

        auto K = mod->get_output_shapes().size() - num_state_vars;
        parameter_map pm;
        std::vector<argument> ret{args.begin(), args.begin() + num_state_vars};
        for(auto i = 0; i < iterations; ++i)
        {
            for(auto j = 0; j < num_state_vars; ++j)
                pm[param_names[j]] = ret[j];
            for(auto j = num_state_vars; j < num_state_vars + K; ++j)
                pm[param_names[j]] = args[i * K + j];

            auto mod_output = run(mod, pm);

            std::copy(mod_output.begin(), mod_output.begin() + num_state_vars, ret.begin());
            ret.insert(ret.end(), mod_output.begin() + num_state_vars, mod_output.end());
        }

        return argument{ret};
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
