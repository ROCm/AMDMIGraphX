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
#include <random>
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
    return ins.get_operator().attributes().contains("target");
}

auto nonprefixed_ops()
{
    // ops without prefixes
    static std::unordered_set<std::string> op_map = {
        "select_module", "load", "if", "nonmaxsuppression", "multibroadcast"};
    return op_map;
}

bool is_compiled_gpu_module(const migraphx::module& m)
{
    return std::all_of(m.begin(), m.end(), [](auto ins) {
        auto ins_name = ins.name();
        if(not migraphx::starts_with(ins_name, "@"))
        {
            if(not migraphx::starts_with(ins_name, "gpu::") and
               not migraphx::starts_with(ins_name, "hip::") and
               not migraphx::starts_with(ins_name, "check_context") and
               not migraphx::contains(nonprefixed_ops(), ins_name) and not has_target_attr(ins))
            {
                return false;
            }
        }
        return true;
    });
}

bool is_compiled_fpga_module(const migraphx::module& m)
{
    return std::all_of(m.begin(), m.end(), [](auto ins) {
        auto ins_name = ins.name();
        if(not migraphx::starts_with(ins_name, "@"))
        {
            if(not migraphx::starts_with(ins_name, "fpga::") and
               not migraphx::starts_with(ins_name, "check_context") and
               not migraphx::contains(nonprefixed_ops(), ins_name) and not has_target_attr(ins))
            {
                return false;
            }
        }
        return true;
    });
}

bool is_compiled_cpu_module(const migraphx::module& m)
{
    return std::all_of(m.begin(), m.end(), [](auto ins) {
        auto ins_name = ins.name();
        if(not migraphx::starts_with(ins_name, "@"))
        {
            if(not migraphx::starts_with(ins_name, "cpu::") and
               not migraphx::starts_with(ins_name, "dnnl::") and
               not migraphx::starts_with(ins_name, "check_context") and not has_target_attr(ins) and
               not migraphx::contains(nonprefixed_ops(), ins_name))
            {
                return false;
            }
        }
        return true;
    });
}

bool is_compiled_ref_module(const migraphx::module& m)
{
    return std::all_of(m.begin(), m.end(), [](auto ins) {
        auto ins_name = ins.name();
        if(not migraphx::starts_with(ins_name, "@"))
        {
            if((not migraphx::starts_with(ins_name, "ref::") and
                not migraphx::starts_with(ins_name, "check_context") and
                not has_target_attr(ins)) and
               not migraphx::contains(nonprefixed_ops(), ins_name))
            {
                return false;
            }
        }
        return true;
    });
}

// NOLINT
bool check_compiled_program(const migraphx::program& p,
                            const std::vector<migraphx::target>& targets)
{
    auto mods           = p.get_modules();
    bool check_compiled = true;
    for(const auto* mod : mods)
    {
        for(const auto& ins : *mod)
        {
            if(ins.name() == "run_on_target")
            {
                auto* mod_input = ins.module_inputs().front();
                std::size_t target_id =
                    ins.get_operator().to_value()["target_id"].to<std::size_t>();
                auto target_name = targets.at(target_id).name();
                if(target_name == "gpu")
                    check_compiled &= is_compiled_gpu_module(*mod_input);
                else if(target_name == "cpu")
                    check_compiled &= is_compiled_cpu_module(*mod_input);
                else if(target_name == "fpga")
                    check_compiled &= is_compiled_fpga_module(*mod_input);
                else if(target_name == "ref")
                    check_compiled &= is_compiled_ref_module(*mod_input);
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
        migraphx::make_op("run_on_target", {{"target_id", 1}}), {x_param, y_param}, {cpu_mod});
    auto gpu_ins = mm->add_instruction(
        migraphx::make_op("run_on_target", {{"target_id", 0}}), {cpu_ins, z_param}, {gpu_mod});
    mm->add_return({gpu_ins});
    p.compile({migraphx::make_target("gpu"), migraphx::make_target("cpu")});
    EXPECT(check_compiled_program(p, {migraphx::make_target("gpu"), migraphx::make_target("cpu")}));
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
    EXPECT(is_compiled_gpu_module(*p.get_main_module()));
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
        migraphx::make_op("run_on_target", {{"target_id", 1}}), {}, {else_mod});
    run_on_cpu_mod->add_return({run_cpu_ins});

    auto* run_on_gpu_mod = p.create_module("run_on_gpu");
    auto run_gpu_ins     = run_on_gpu_mod->add_instruction(
        migraphx::make_op("run_on_target", {{"target_id", 0}}), {}, {then_mod});
    run_on_gpu_mod->add_return({run_gpu_ins});

    auto ret =
        mm->add_instruction(migraphx::make_op("if"), {cond}, {run_on_gpu_mod, run_on_cpu_mod});
    auto r = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), ret);
    mm->add_return({r});
    // compile
    p.compile({migraphx::make_target("gpu"), migraphx::make_target("cpu")});
    EXPECT(check_compiled_program(p, {migraphx::make_target("gpu"), migraphx::make_target("cpu")}));
}

TEST_CASE(multitarget_compile_nested_if_then_else)
{
    float seed = 0.0f;
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    auto get_random_values = [&](size_t elements) {
        std::vector<float> rand_samples(elements);
        std::generate(rand_samples.begin(), rand_samples.end(), [&]() { return dis(gen); });
        return rand_samples;
    };

    std::unordered_map<std::size_t, std::size_t> counter_map = {{0, 0}, {1, 0}};
    migraphx::shape ds{migraphx::shape::float_type, {2, 3}};
    migraphx::program p;
    auto* mm = p.get_main_module();
    migraphx::shape cond_s{migraphx::shape::bool_type};
    auto cond               = mm->add_parameter("cond", cond_s);
    auto x                  = mm->add_parameter("x", ds);
    auto y                  = mm->add_parameter("y", ds);
    auto z                  = mm->add_parameter("z", ds);
    auto create_test_module = [&](migraphx::program& prog,
                                  const std::vector<migraphx::instruction_ref>& inputs,
                                  std::size_t tid) {
        std::string mod_name =
            "target_" + std::to_string(tid) + "_" + std::to_string(counter_map[tid]++);
        auto* test_mod          = prog.create_module(mod_name);
        std::vector<float> data = get_random_values(ds.elements());
        auto l1                 = test_mod->add_literal(migraphx::literal(ds, data));
        auto test_mod_param     = test_mod->add_parameter(mod_name, ds);
        // instruction with local literal and main_mod param as inputs
        auto ins1 = test_mod->add_instruction(migraphx::make_op("add"), x, l1);
        // instructinon with local param and local ins as inputs
        auto ins2 = test_mod->add_instruction(migraphx::make_op("mul"), ins1, test_mod_param);
        // instruction with local ins and parent ins as inputs
        auto ins3 = test_mod->add_instruction(migraphx::make_op("sub"), ins2, inputs.front());
        test_mod->add_return({ins3});
        auto* run_on_target_mod = prog.create_module("run_on_" + mod_name);
        run_on_target_mod->add_instruction(
            migraphx::make_op("run_on_target", {{"target_id", tid}}), {inputs.front()}, {test_mod});
        return run_on_target_mod;
    };

    // create nested module with multiple targets.
    // then_mod has one instruction that runs a module on "ref" and another instruction that
    // creates nested modules using "If" that runs on "cpu" and "gpu"
    auto* ref_mod = p.create_module("ref_mod");
    auto ref_x    = ref_mod->add_parameter("ref_x", ds);
    auto ref_y    = ref_mod->add_parameter("ref_y", ds);
    auto ref_add  = ref_mod->add_instruction(migraphx::make_op("add"), ref_x, ref_y);
    ref_mod->add_return({ref_add});

    auto* then_mod        = p.create_module("then_mod");
    auto then_mod_param   = then_mod->add_parameter("then_mod_param", ds);
    auto then_mod_ref_ins = then_mod->add_instruction(
        migraphx::make_op("run_on_target", {{"target_id", 3}}), {then_mod_param, y}, {ref_mod});
    auto then_mod_ref_ins_0 = then_mod->add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), then_mod_ref_ins);
    then_mod->add_instruction(
        migraphx::make_op("if"),
        {cond},
        {create_test_module(p, {z}, 1), create_test_module(p, {then_mod_ref_ins_0}, 0)});

    // create nested else_mod with multiple targets.
    // else_mod has one instruction that runs a module on "fpga" and another instruction that
    // creates nested modules using "If" that runs on "cpu" and "gpu"
    auto* fpga_mod = p.create_module("fpga_mod");
    auto fpga_x    = fpga_mod->add_parameter("fpga_x", ds);
    auto fpga_y    = fpga_mod->add_parameter("fpga_y", ds);
    auto fpga_add  = fpga_mod->add_instruction(migraphx::make_op("add"), fpga_x, fpga_y);
    fpga_mod->add_return({fpga_add});

    auto* else_mod         = p.create_module("else_mod");
    auto else_mod_param    = else_mod->add_parameter("else_mod_param", ds);
    auto else_mod_fpga_ins = else_mod->add_instruction(
        migraphx::make_op("run_on_target", {{"target_id", 2}}), {else_mod_param, y}, {fpga_mod});
    auto else_mod_fpga_ins_0 = else_mod->add_instruction(
        migraphx::make_op("get_tuple_elem", {{"index", 0}}), else_mod_fpga_ins);

    else_mod->add_instruction(migraphx::make_op("if"),
                              {cond},
                              {create_test_module(p, {else_mod_fpga_ins_0}, 0),
                               create_test_module(p, {else_mod_param}, 1)});

    // Create nested and multi-target main module using "If"
    auto main_if_ins =
        mm->add_instruction(migraphx::make_op("if"), {cond, x}, {then_mod, else_mod});
    auto r = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), main_if_ins);
    mm->add_return({r});

    // compile
    p.compile({migraphx::make_target("gpu"),
               migraphx::make_target("cpu"),
               migraphx::make_target("fpga"),
               migraphx::make_target("ref")});
    EXPECT(check_compiled_program(p,
                                  {migraphx::make_target("gpu"),
                                   migraphx::make_target("cpu"),
                                   migraphx::make_target("fpga"),
                                   migraphx::make_target("ref")}));
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
    auto input        = mm->add_parameter("data", s);
    auto* run_cpu_mod = p.create_module("cpu_mod");
    auto cpu_param    = run_cpu_mod->add_parameter(
        "cpu_data", migraphx::shape{migraphx::shape::float_type, {1, 4}});
    auto run_cpu_ins = run_cpu_mod->add_instruction(
        migraphx::make_op("run_on_target", {{"target_id", 1}}), {cpu_param}, {batch1});
    run_cpu_mod->add_return({run_cpu_ins});

    auto* run_gpu_mod = p.create_module("gpu_mod");
    auto gpu_param    = run_gpu_mod->add_parameter(
        "gpu_data", migraphx::shape{migraphx::shape::float_type, {2, 4}});
    auto run_gpu_ins = run_gpu_mod->add_instruction(
        migraphx::make_op("run_on_target", {{"target_id", 0}}), {gpu_param}, {batch2});
    run_gpu_mod->add_return({run_gpu_ins});

    auto* run_fpga_mod = p.create_module("fpga_mod");
    auto fpga_param    = run_fpga_mod->add_parameter(
        "fpga_data", migraphx::shape{migraphx::shape::float_type, {3, 4}});
    auto run_fpga_ins = run_fpga_mod->add_instruction(
        migraphx::make_op("run_on_target", {{"target_id", 2}}), {fpga_param}, {batch3});
    run_fpga_mod->add_return({run_fpga_ins});

    auto* run_ref_mod = p.create_module("ref_mod");
    auto ref_param    = run_fpga_mod->add_parameter(
        "ref_data", migraphx::shape{migraphx::shape::float_type, {4, 4}});
    auto run_ref_ins = run_ref_mod->add_instruction(
        migraphx::make_op("run_on_target", {{"target_id", 3}}), {ref_param}, {batch4});
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
    p.compile({migraphx::make_target("gpu"),
               migraphx::make_target("cpu"),
               migraphx::make_target("fpga"),
               migraphx::make_target("ref")});
    EXPECT(check_compiled_program(p,
                                  {migraphx::make_target("gpu"),
                                   migraphx::make_target("cpu"),
                                   migraphx::make_target("fpga"),
                                   migraphx::make_target("ref")}));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
