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
#include "make_precompile_op.hpp"
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/fuse_ops.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>
#include <test.hpp>
#include <pointwise.hpp>

void run_pass(migraphx::program& p)
{
    migraphx::run_passes(p, {migraphx::gpu::fuse_ops{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(layernorm_pointwise)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};
    auto create_program = [=](bool first_arg_layernorm) {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto x         = mm->add_parameter("x", s);
        auto y         = mm->add_parameter("y", s);
        auto z         = mm->add_parameter("z", s);
        auto alloc     = migraphx::make_op("allocate", {{"shape", to_value(s)}});
        auto alloc_ins = mm->add_instruction(alloc);
        auto* pw_add1 =
            create_pointwise_module(p, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto add1 =
            mm->add_instruction(make_precompile_op("pointwise"), {x, y, alloc_ins}, {pw_add1});
        auto alloc_ins2 = mm->add_instruction(alloc);
        auto layernorm_ins =
            mm->add_instruction(make_precompile_op("gpu::prelayernorm"), add1, alloc_ins2);
        std::vector<migraphx::instruction_ref> pw_inputs = {layernorm_ins, z};
        if(not first_arg_layernorm)
        {
            pw_inputs = {z, layernorm_ins};
        }
        auto* pw_add2 =
            create_pointwise_module(p, "main:pointwise1", pw_inputs, single_pointwise("add"));
        auto alloc_ins3 = mm->add_instruction(alloc);
        pw_inputs.push_back(alloc_ins3);
        auto add2 = mm->add_instruction(make_precompile_op("pointwise"), pw_inputs, {pw_add2});
        mm->add_return({add2});
        return p;
    };

    auto create_fused_program = [=]() {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto x         = mm->add_parameter("x", s);
        auto y         = mm->add_parameter("y", s);
        auto z         = mm->add_parameter("z", s);
        auto alloc     = migraphx::make_op("allocate", {{"shape", to_value(s)}});
        auto alloc_ins = mm->add_instruction(alloc);
        auto* pw_add1 =
            create_pointwise_module(p, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto add1 =
            mm->add_instruction(make_precompile_op("pointwise"), {x, y, alloc_ins}, {pw_add1});
        auto alloc_ins2 = mm->add_instruction(alloc);
        auto* pw_add2 =
            create_pointwise_module(p, "main:pointwise1", {x, z}, single_pointwise("add"));
        auto layernorm_ins = mm->add_instruction(
            make_precompile_op("gpu::prelayernorm"), {add1, z, alloc_ins2}, {pw_add2});
        mm->add_return({layernorm_ins});
        return p;
    };

    {
        migraphx::program p1 = create_program(true);
        run_pass(p1);
        migraphx::program p2 = create_fused_program();
        EXPECT(p1 == p2);
    }
    {
        migraphx::program p1 = create_program(false);
        run_pass(p1);
        migraphx::program p2 = create_fused_program();
        EXPECT(p1 == p2);
    }
}

TEST_CASE(contiguous_pointwise)
{
    migraphx::shape s1{migraphx::shape::float_type, {128, 4, 196, 32}};
    migraphx::shape s2{migraphx::shape::float_type, {128, 196, 4, 32}};

    auto create_program = [=]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto x_trans =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), x);
        auto alloc     = migraphx::make_op("allocate", {{"shape", to_value(s2)}});
        auto alloc_ins = mm->add_instruction(alloc);
        auto* pw_add1 =
            create_pointwise_module(p, "main:pointwise0", {x_trans, y}, single_pointwise("add"));
        auto add1 = mm->add_instruction(
            make_precompile_op("pointwise"), {x_trans, y, alloc_ins}, {pw_add1});

        auto alloc_ins2 = mm->add_instruction(alloc);
        auto cont = mm->add_instruction(migraphx::make_op("gpu::contiguous"), add1, alloc_ins2);
        auto rsp =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {25088, 128}}}), cont);
        mm->add_return({rsp});
        return p;
    };

    auto create_fused_program = [=]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto x_trans =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3}}}), x);
        auto alloc     = migraphx::make_op("allocate", {{"shape", to_value(s2)}});
        auto alloc_ins = mm->add_instruction(alloc);
        auto* pw_add1 =
            create_pointwise_module(p, "main:pointwise0", {x_trans, y}, single_pointwise("add"));

        auto pw_op       = migraphx::make_op("pointwise");
        auto pre_comp_op = migraphx::make_op(
            "gpu::precompile_op",
            {{"op", migraphx::to_value(pw_op)}, {"output_shape", migraphx::to_value(s2)}});
        auto add1 = mm->add_instruction(pre_comp_op, {x_trans, y, alloc_ins}, {pw_add1});
        auto rsp =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {25088, 128}}}), add1);
        mm->add_return({rsp});
        return p;
    };

    migraphx::program p1 = create_program();
    run_pass(p1);
    migraphx::program p2 = create_fused_program();
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
