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
#include <unordered_set>
#include <vector>
#include <cmath>
#include <migraphx/program.hpp>
#include <migraphx/target.hpp>
#include <migraphx/stringutils.hpp>
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

// check if it is custom_op or run_on_module operator
bool has_target_attr(const migraphx::instruction& ins)
{
    if(ins.get_operator().attributes().contains("target"))
    {
        return true;
    }
    return false;
}

auto nonprefixed_ops()
{
    // ops without prefixes
    static std::unordered_set<std::string> op_map = {
        "select_module", "load", "if", "nonmaxsuppression"};
    return op_map;
}

bool is_compiled_gpu_module(const migraphx::module& m)
{
    for(auto ins : m)
    {
        auto ins_name = ins.name();
        if(!migraphx::starts_with(ins_name, "@"))
        {
            if(!migraphx::starts_with(ins_name, "gpu::") and
               !migraphx::starts_with(ins_name, "hip::") and
               !migraphx::starts_with(ins_name, "check_context") and
               !migraphx::contains(nonprefixed_ops(), ins_name) and !has_target_attr(ins))
            {
                return false;
            }
        }
        else
        {
            continue;
        }
    }
    return true;
}

bool is_compiled_fpga_module(const migraphx::module& m)
{
    for(auto ins : m)
    {
        auto ins_name = ins.name();
        if(!migraphx::starts_with(ins_name, "@"))
        {
            if(!migraphx::starts_with(ins_name, "fpga::") and
               !migraphx::starts_with(ins_name, "check_context") and
               !migraphx::contains(nonprefixed_ops(), ins_name) and !has_target_attr(ins))
            {
                return false;
            }
        }
        else
        {
            continue;
        }
    }
    return true;
}

bool is_compiled_cpu_module(const migraphx::module& m)
{
    for(auto ins : m)
    {
        auto ins_name = ins.name();
        if(!migraphx::starts_with(ins_name, "@"))
        {
            if(!migraphx::starts_with(ins_name, "cpu::") and
               !migraphx::starts_with(ins_name, "dnnl::") and
               !migraphx::starts_with(ins_name, "check_context") and !has_target_attr(ins) and
               !migraphx::contains(nonprefixed_ops(), ins_name))
            {
                return false;
            }
        }
        else
        {
            continue;
        }
    }
    return true;
}

bool has_gpu_copies(const migraphx::module& m)
{
    bool hip_copy_from_gpu = false;
    bool hip_copy_to_gpu   = false;
    for(const auto& ins : m)
    {
        auto ins_name = ins.name();
        if(ins_name == "hip::copy_from_gpu")
        {
            hip_copy_from_gpu = true;
        }
        else if(ins_name == "hip::copy_to_gpu")
        {
            hip_copy_to_gpu = true;
        }
    }
    return hip_copy_to_gpu and hip_copy_from_gpu;
}

bool is_compiled_ref_module(const migraphx::module& m)
{
    for(auto ins : m)
    {
        auto ins_name = ins.name();
        if(!migraphx::starts_with(ins_name, "@"))
        {
            if((!migraphx::starts_with(ins_name, "ref::") and
                !migraphx::starts_with(ins_name, "check_context") and !has_target_attr(ins)) and
               !migraphx::contains(nonprefixed_ops(), ins_name))
            {
                return false;
            }
        }
        else
        {
            continue;
        }
    }
    return true;
}

bool check_compiled_program(const migraphx::program& p,
                            std::unordered_map<std::string, migraphx::compile_options> copts)
{
    auto mods           = p.get_modules();
    bool check_compiled = true;
    for(const auto* mod : mods)
    {
        for(const auto& ins : *mod)
        {
            if(ins.name() == "run_on_target")
            {
                auto mod_input   = ins.module_inputs().front();
                auto target_name = ins.get_operator().attributes()["target"];
                if(target_name == "gpu")
                {
                    check_compiled &= is_compiled_gpu_module(*mod_input);
                    if(contains(copts, "gpu"))
                    {
                        if(copts["gpu"].offload_copy)
                        {
                            check_compiled &= has_gpu_copies(*mod_input);
                        }
                    }
                }
                else if(target_name == "cpu")
                {
                    check_compiled &= is_compiled_cpu_module(*mod_input);
                }
                else if(target_name == "fpga")
                {
                    check_compiled &= is_compiled_fpga_module(*mod_input);
                }
                else if(target_name == "ref")
                {
                    check_compiled &= is_compiled_ref_module(*mod_input);
                }
            }
        }
    }

    return check_compiled;
}

TEST_CASE(multitarget_compile_cpu_gpu)
{
    migraphx::program p;
    auto* mm      = p.get_main_module();
    auto* cpu_mod = p.create_module("cpu_mod");
    auto s        = migraphx::shape{migraphx::shape::float_type, {8}};
    auto x_cpu    = cpu_mod->add_parameter("cpu_x", s);
    auto y_cpu    = cpu_mod->add_parameter("cpu_y", s);
    auto cpu_add  = cpu_mod->add_instruction(migraphx::make_op("add"), x_cpu, y_cpu);
    cpu_mod->add_return({cpu_add});

    auto* gpu_mod = p.create_module("gpu_mod");
    auto x_gpu    = gpu_mod->add_parameter("gpu_x", s);
    auto y_gpu    = gpu_mod->add_parameter("gpu_y", s);
    auto gpu_add  = gpu_mod->add_instruction(migraphx::make_op("add"), x_gpu, y_gpu);
    gpu_mod->add_return({gpu_add});

    auto x_param = mm->add_parameter("x", s);
    auto y_param = mm->add_parameter("y", s);
    auto z_param = mm->add_parameter("z", s);
    auto cpu_ins = mm->add_instruction(
        migraphx::make_op("run_on_target", {{"target", "cpu"}}), {x_param, y_param}, {cpu_mod});
    auto gpu_ins = mm->add_instruction(
        migraphx::make_op("run_on_target", {{"target", "gpu"}}), {cpu_ins, z_param}, {gpu_mod});
    mm->add_return({gpu_ins});
    std::unordered_map<std::string, migraphx::compile_options> compile_opts;
    migraphx::compile_options gpu_opt;
    gpu_opt.offload_copy = true;
    compile_opts["gpu"]  = gpu_opt;
    p.compile({"gpu", "cpu"}, compile_opts);
    CHECK(check_compiled_program(p, compile_opts));
}

TEST_CASE(single_target_compile)
{
    migraphx::program p;
    auto* mm = p.get_main_module();

    migraphx::shape boxes_s{migraphx::shape::float_type, {1, 6, 4}};

    migraphx::shape scores_s{migraphx::shape::float_type, {1, 1, 6}};
    std::vector<float> scores_vec = {0.9, 0.75, 0.6, 0.95, 0.5, 0.3};

    auto boxes_l         = mm->add_parameter("boxes", boxes_s);
    auto scores_l        = mm->add_literal(migraphx::literal(scores_s, scores_vec));
    auto max_out_l       = mm->add_literal(int64_t{4});
    auto iou_threshold   = mm->add_literal(0.5f);
    auto score_threshold = mm->add_literal(0.0f);

    auto r = mm->add_instruction(migraphx::make_op("nonmaxsuppression", {{"center_point_box", 1}}),
                                 boxes_l,
                                 scores_l,
                                 max_out_l,
                                 iou_threshold,
                                 score_threshold);
    mm->add_return({r});
    p.compile(migraphx::make_target("gpu"));
    CHECK(is_compiled_gpu_module(*p.get_main_module()));
}

TEST_CASE(multitarget_compile_if_then_else)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape cond_s{migraphx::shape::bool_type};
    auto cond = mm->add_parameter("cond", cond_s);
    migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
    auto x = mm->add_parameter("x", ds);
    auto y = mm->add_parameter("y", ds);

    auto* then_mod           = p.create_module("if_gpu_mod");
    std::vector<float> data1 = {0.384804, -1.77948, -0.453775, 0.477438, -1.06333, -1.12893};
    auto l1                  = then_mod->add_literal(migraphx::literal(ds, data1));
    auto a1                  = then_mod->add_instruction(migraphx::make_op("add"), x, l1);
    then_mod->add_return({a1});

    auto* else_mod           = p.create_module("else_cpu_mod");
    std::vector<float> data2 = {-0.258047, 0.360394, 0.536804, -0.577762, 1.0217, 1.02442};
    auto l2                  = else_mod->add_literal(migraphx::literal(ds, data2));
    auto a2                  = else_mod->add_instruction(migraphx::make_op("mul"), y, l2);
    else_mod->add_return({a2});

    auto* run_on_cpu_mod = p.create_module("run_on_cpu");
    auto run_cpu_ins     = run_on_cpu_mod->add_instruction(
        migraphx::make_op("run_on_target", {{"target", "cpu"}}), {}, {else_mod});
    run_on_cpu_mod->add_return({run_cpu_ins});

    auto* run_on_gpu_mod = p.create_module("run_on_gpu");
    auto run_gpu_ins     = run_on_gpu_mod->add_instruction(
        migraphx::make_op("run_on_target", {{"target", "gpu"}}), {}, {then_mod});
    run_on_gpu_mod->add_return({run_gpu_ins});

    auto ret =
        mm->add_instruction(migraphx::make_op("if"), {cond}, {run_on_gpu_mod, run_on_cpu_mod});
    auto r = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
    mm->add_return({r});
    // compile
    std::unordered_map<std::string, migraphx::compile_options> compile_opts;
    migraphx::compile_options gpu_opt;
    gpu_opt.offload_copy = true;
    compile_opts["gpu"]  = gpu_opt;
    p.compile(
        {
            "gpu",
            "cpu",
        },
        compile_opts);
    p.debug_print();
    CHECK(check_compiled_program(p, compile_opts));
}

TEST_CASE(multitarget_select_module)
{
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape lit_s{migraphx::shape{migraphx::shape::float_type, {1}}};
    auto literal_ins = mm->add_literal(migraphx::literal{lit_s, {6}});

    // create batch submodules
    auto create_submodule = [&](std::size_t batch_size, const std::string& module_name) {
        auto* submod = p.create_module(module_name);
        migraphx::shape sm_shape{migraphx::shape::float_type, {batch_size, 4}};
        auto sm_input = submod->add_parameter("data", sm_shape);
        auto broadcast_lit =
            submod->add_instruction(migraphx::make_op("multibroadcast"), literal_ins, sm_input);
        auto add_ins0 = submod->add_instruction(migraphx::make_op("add"), sm_input, broadcast_lit);
        auto add_ins1 = submod->add_instruction(migraphx::make_op("add"), add_ins0, broadcast_lit);
        submod->add_return({add_ins0, add_ins1});
        return submod;
    };
    auto* batch1 = create_submodule(1, "batch_1");
    auto* batch2 = create_submodule(2, "batch_2");
    auto* batch3 = create_submodule(3, "batch_3");
    auto* batch4 = create_submodule(4, "batch_4");

    migraphx::shape s{migraphx::shape::float_type, {{1, 4}, {4, 4}}};
    auto input       = mm->add_parameter("data", s);
    auto run_cpu_mod = p.create_module("cpu_mod");
    auto run_cpu_ins = run_cpu_mod->add_instruction(
        migraphx::make_op("run_on_target", {{"target", "cpu"}}), {input}, {batch1});
    run_cpu_mod->add_return({run_cpu_ins});

    auto run_gpu_mod = p.create_module("gpu_mod");
    auto run_gpu_ins = run_gpu_mod->add_instruction(
        migraphx::make_op("run_on_target", {{"target", "gpu"}}), {input}, {batch2});
    run_gpu_mod->add_return({run_gpu_ins});

    auto run_fpga_mod = p.create_module("fpga_mod");
    auto run_fpga_ins = run_fpga_mod->add_instruction(
        migraphx::make_op("run_on_target", {{"target", "fpga"}}), {input}, {batch3});
    run_fpga_mod->add_return({run_fpga_ins});

    auto run_ref_mod = p.create_module("ref_mod");
    auto run_ref_ins = run_ref_mod->add_instruction(
        migraphx::make_op("run_on_target", {{"target", "ref"}}), {input}, {batch4});
    run_ref_mod->add_return({run_ref_ins});

    std::vector<migraphx::shape> sub_shapes = {};
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
    sub_shapes.push_back(migraphx::shape{migraphx::shape::float_type, {{1, 4}, {4, 4}}});
    migraphx::shape out_attr = migraphx::shape{sub_shapes};
    auto sm_ins              = mm->add_instruction(
        migraphx::make_op("select_module", {{"output_dyn_shapes", migraphx::to_value(out_attr)}}),
        {input},
        {run_cpu_mod, run_gpu_mod, run_fpga_mod, run_ref_mod});
    auto ret0 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), sm_ins);
    auto ret1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), sm_ins);
    mm->add_return({ret0, ret1});
    // compile
    std::unordered_map<std::string, migraphx::compile_options> compile_opts;
    migraphx::compile_options gpu_opt;
    gpu_opt.offload_copy = true;
    compile_opts["gpu"]  = gpu_opt;
    p.compile({"gpu", "cpu", "fpga", "ref"}, compile_opts);
    CHECK(check_compiled_program(p, compile_opts));
}

TEST_CASE(multitarget_compile_nested_modules) {}

TEST_CASE(mulitarget_compile_dead_code_elimination) {}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
