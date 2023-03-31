/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <iostream>
#include <set>
#include <vector>
#include <cmath>
#include <migraphx/program.hpp>
#include <migraphx/target.hpp>
#include <migraphx/module.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/verify.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/check_shapes.hpp>
#include <migraphx/functional.hpp>
#include <basic_ops.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/register_target.hpp>
#include "test.hpp"

struct mod_run_op
{
    std::string name() const { return "mod_run_op"; }

    migraphx::shape compute_shape(std::vector<migraphx::shape> inputs,
                                  std::vector<migraphx::module_ref> mods) const
    {
        if(not mods.empty())
        {
            auto out_shapes = mods[0]->get_output_shapes();
            return out_shapes[0];
        }
        if(not inputs.empty())
        {
            return inputs.front();
        }
        return {};
    }

    migraphx::argument
    compute(const migraphx::shape&,
            const std::vector<migraphx::argument>& args,
            const std::vector<migraphx::module_ref>& mods,
            const std::function<std::vector<migraphx::argument>(
                migraphx::module_ref&, const std::unordered_map<std::string, migraphx::argument>&)>&
                run) const

    {
        std::unordered_map<std::string, migraphx::argument> params;
        std::set<std::string> pnames;
        for(const auto& smod : mods)
        {
            smod->debug_print();
            auto names = smod->get_parameter_names();
            pnames.insert(names.begin(), names.end());
        }

        assert(pnames.size() <= args.size());
        std::transform(pnames.begin(),
                       pnames.end(),
                       args.begin(),
                       std::inserter(params, params.end()),
                       [](auto&& name, auto&& arg) { return std::make_pair(name, arg); });
        auto mod     = mods[0];
        auto results = run(mod, params);
        return results.front();
    }
};

TEST_CASE(multitarget_program)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    mm->set_target("ref");
    auto* cpu_mod = p.create_module("cpu_mod");
    cpu_mod->set_target("cpu");
    auto s       = migraphx::shape{migraphx::shape::float_type, {8}};
    auto x_cpu   = cpu_mod->add_parameter("cpu_x", s);
    auto y_cpu   = cpu_mod->add_parameter("cpu_y", s);
    auto cpu_add = cpu_mod->add_instruction(migraphx::make_op("add"), x_cpu, y_cpu);
    cpu_mod->add_return({cpu_add});

    auto* gpu_mod = p.create_module("gpu_mod");
    gpu_mod->set_target("gpu");
    migraphx::compile_options gpu_opt;
    gpu_opt.offload_copy = true;
    auto x_gpu           = gpu_mod->add_parameter("gpu_x", s);
    auto y_gpu           = gpu_mod->add_parameter("gpu_y", s);
    auto gpu_add         = gpu_mod->add_instruction(migraphx::make_op("add"), x_gpu, y_gpu);
    gpu_mod->add_return({gpu_add});

    auto x_param = mm->add_parameter("x", s);
    auto y_param = mm->add_parameter("y", s);
    auto z_param = mm->add_parameter("z", s);
    auto cpu_ins = mm->add_instruction(mod_run_op{}, {x_param, y_param}, {cpu_mod});
    auto gpu_ins = mm->add_instruction(mod_run_op{}, {cpu_ins, z_param}, {gpu_mod});
    mm->add_return({gpu_ins});
    (void)migraphx::make_target("gpu");
    (void)migraphx::make_target("cpu");
    std::unordered_map<std::string, migraphx::compile_options> compile_opts;
    compile_opts["gpu"] = gpu_opt;
    std::cout << "program before compile\n";
    p.debug_print();
    std::cout << "===============================\n";
    p.multitarget_compile(compile_opts);
    std::cout << "Compiled program \n";
    p.debug_print();
    migraphx::parameter_map params_map;
    std::vector<float> x_data = {1, 1, 1, 1, 1, 1, 1, 1};
    params_map["x"]           = migraphx::argument(s, x_data.data());
    params_map["y"]           = migraphx::argument(s, x_data.data());
    params_map["z"]           = migraphx::argument(s, x_data.data());

    auto result = p.eval(params_map).back();
    std::vector<float> results_vector(8, -1);
    result.visit([&](auto output) { results_vector.assign(output.begin(), output.end()); });
    std::vector<float> gold = {3, 3, 3, 3, 3, 3, 3, 3};
    EXPECT(migraphx::verify_range(results_vector, gold));
    std::cout << "Success\n";
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
