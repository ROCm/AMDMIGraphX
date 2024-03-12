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
        std::cout << "SCAN COMPUTE SHAPE" << std::endl;
        assert(mods.size() == 1);
        check_shapes{inputs, *this}.standard();
        auto mod = mods.front();
        std::cout << "Inputs size: " << inputs.size() << std::endl;
        // The module has:
        // N + M inputs (M = num_scan_inputs), same as the Scan node itself
        // N + K outputs, same as the Scan node itself
        auto output_shapes = mod->get_output_shapes();
        std::cout << to_string_range(output_shapes) << std::endl;
        // Can't use mod->get_parameter_shapes() like this, parameters can include output parameters
        // as well
        // auto N = mod->get_parameter_shapes().size() - num_scan_inputs;
        auto N = num_state_vars;
        std::transform(output_shapes.begin() + N,
                       output_shapes.end(),
                       output_shapes.begin() + N,
                       [&](const auto& s) {
                           auto lens = s.lens();
                           lens.insert(lens.begin(), iterations);
                           return shape{s.type(), lens};
                       });

        std::cout << "OUTPUT SHAPES: " << std::endl;
        std::cout << to_string_range(output_shapes) << std::endl;

        return shape{output_shapes};
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
        std::cout << "SCAN COMPUTE" << std::endl;
        std::cout << args.size() << std::endl;
        std::cout << out_shape << std::endl;
        assert(mods.size() == 1);
        auto mod          = mods.front();
        mod->debug_print();
        auto param_shapes = mod->get_parameter_shapes();
        for(const auto& s : param_shapes)
            std::cout << s.first << " " << s.second << std::endl;
        auto output_params = get_output_params(mod);
        for(const auto& p : output_params)
            std::cout << p.first << " " << p.second << std::endl;
        for(int64_t i = 0; i < iterations; ++i)
        {
            // Prepare params
            // Run 
            // Set up next iteration
        }
        return args[0];
    }
};

} // namespace op
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
