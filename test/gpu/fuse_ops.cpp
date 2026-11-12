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

static void run_pass(migraphx::program& p)
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
        auto layernorm_op = migraphx::make_op("gpu::prelayernorm");
        auto pre_comp_op  = migraphx::make_op(
            "gpu::precompile_op",
            {{"op", migraphx::to_value(layernorm_op)}, {"output_shape", migraphx::to_value(s)}});

        auto layernorm_ins = mm->add_instruction(pre_comp_op, {add1, z, alloc_ins2}, {pw_add2});
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
        EXPECT(p1 == create_program(false));
    }
}

TEST_CASE(pointwise_contiguous)
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

TEST_CASE(layout_pointwise)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 8, 4, 4}, {128, 1, 32, 8}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 8, 4, 4}};

    auto create_program = [=]() {
        migraphx::program p;
        auto* mm        = p.get_main_module();
        auto x          = mm->add_parameter("x", s2);
        auto y          = mm->add_parameter("y", s2);
        auto layout_op  = migraphx::make_op("layout", {{"permutation", {0, 2, 3, 1}}});
        auto alloc1     = migraphx::make_op("allocate", {{"shape", to_value(s1)}});
        auto alloc_ins1 = mm->add_instruction(alloc1);
        auto x_nhwc     = mm->add_instruction(
            migraphx::make_op("gpu::precompile_op", {{"op", migraphx::to_value(layout_op)}}),
            x,
            alloc_ins1);
        auto alloc2     = migraphx::make_op("allocate", {{"shape", to_value(s2)}});
        auto alloc_ins2 = mm->add_instruction(alloc2);
        auto* pw_add =
            create_pointwise_module(p, "main:pointwise0", {x_nhwc, y}, single_pointwise("add"));
        auto add =
            mm->add_instruction(make_precompile_op("pointwise"), {x_nhwc, y, alloc_ins2}, {pw_add});
        auto rsp =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {1, 8, 16}}}), add);
        mm->add_return({rsp});
        return p;
    };

    auto create_fused_program = [=]() {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto x         = mm->add_parameter("x", s2);
        auto y         = mm->add_parameter("y", s2);
        auto alloc     = migraphx::make_op("allocate", {{"shape", to_value(s2)}});
        auto alloc_ins = mm->add_instruction(alloc);
        auto* pw_add =
            create_pointwise_module(p, "main:pointwise0", {x, y}, single_pointwise("add"));

        auto pw_op       = migraphx::make_op("pointwise");
        auto pre_comp_op = migraphx::make_op(
            "gpu::precompile_op",
            {{"op", migraphx::to_value(pw_op)}, {"output_shape", migraphx::to_value(s2)}});
        auto add = mm->add_instruction(pre_comp_op, {x, y, alloc_ins}, {pw_add});
        auto rsp =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {1, 8, 16}}}), add);
        mm->add_return({rsp});
        return p;
    };

    migraphx::program p1 = create_program();
    run_pass(p1);
    migraphx::program p2 = create_fused_program();
    EXPECT(p1 == p2);
}

TEST_CASE(contiguous_pointwise)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 4, 4, 8}};
    migraphx::shape s2{migraphx::shape::float_type, {1, 8, 4, 4}};

    auto create_program = [=]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto x_trans =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), x);
        auto alloc      = migraphx::make_op("allocate", {{"shape", to_value(s2)}});
        auto alloc_ins1 = mm->add_instruction(alloc);
        auto x_cont =
            mm->add_instruction(migraphx::make_op("gpu::contiguous"), x_trans, alloc_ins1);
        auto alloc_ins2 = mm->add_instruction(alloc);
        auto* pw_add =
            create_pointwise_module(p, "main:pointwise0", {x_cont, y}, single_pointwise("add"));
        auto add =
            mm->add_instruction(make_precompile_op("pointwise"), {x_cont, y, alloc_ins2}, {pw_add});
        auto rsp =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {1, 8, 16}}}), add);
        mm->add_return({rsp});
        return p;
    };

    auto create_fused_program = [=]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto x_trans =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 3, 1, 2}}}), x);
        auto alloc     = migraphx::make_op("allocate", {{"shape", to_value(s2)}});
        auto alloc_ins = mm->add_instruction(alloc);
        auto* pw_add =
            create_pointwise_module(p, "main:pointwise0", {x_trans, y}, single_pointwise("add"));

        auto pw_op       = migraphx::make_op("pointwise");
        auto pre_comp_op = migraphx::make_op(
            "gpu::precompile_op",
            {{"op", migraphx::to_value(pw_op)}, {"output_shape", migraphx::to_value(s2)}});
        auto add = mm->add_instruction(pre_comp_op, {x_trans, y, alloc_ins}, {pw_add});
        auto rsp =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {1, 8, 16}}}), add);
        mm->add_return({rsp});
        return p;
    };
}

// gpu::convolution not supported since MIOpen is OFF
#if MIGRAPHX_USE_MIOPEN
TEST_CASE(pointwise_layout_convolution)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 320, 128, 128}};
    migraphx::shape s2{migraphx::shape::float_type, {320, 320, 3, 3}, {2880, 1, 960, 320}};
    migraphx::shape s3{migraphx::shape::float_type, {2, 320, 128, 128}, {5242880, 1, 40960, 320}};
    // workspace for gpu::convolution, memory space can change based on gfx arch and rocm version,
    // For the unit-test just use some random number.
    migraphx::shape s4{migraphx::shape::int8_type, {41943040}};

    auto create_program = [=]() {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto x1        = mm->add_parameter("x1", s1);
        auto x2        = mm->add_parameter("x2", s1);
        auto weights   = mm->add_parameter("weights", s2);
        auto alloc     = migraphx::make_op("allocate", {{"shape", to_value(s1)}});
        auto alloc_ins = mm->add_instruction(alloc);
        auto* pwm      = create_pointwise_module(
            p, "main:pointwise0", {x1, x2}, [=](auto* pm, const auto& inputs) {
                auto mul_ins = pm->add_instruction(migraphx::make_op("mul"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("sigmoid"), mul_ins);
            });
        auto pw_ins =
            mm->add_instruction(make_precompile_op("pointwise"), {x1, x2, alloc_ins}, {pwm});

        auto alloc_ins2 =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(s3)}}));
        auto layout_op  = migraphx::make_op("layout", {{"permutation", {0, 2, 3, 1}}});
        auto layout_ins = mm->add_instruction(make_precompile_op(layout_op), {pw_ins, alloc_ins2});
        auto conv_op    = migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}});
        auto alloc_ins3 =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(s4)}}));
        auto alloc_ins4 =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(s3)}}));
        auto conv =
            mm->add_instruction(migraphx::make_op("gpu::convolution", {{"op", conv_op.to_value()}}),
                                layout_ins,
                                weights,
                                alloc_ins3,
                                alloc_ins4);
        mm->add_return({conv});
        return p;
    };
    auto create_fused_program = [=]() {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto x1        = mm->add_parameter("x1", s1);
        auto x2        = mm->add_parameter("x2", s1);
        auto weights   = mm->add_parameter("weights", s2);
        auto alloc     = migraphx::make_op("allocate", {{"shape", to_value(s3)}});
        auto alloc_ins = mm->add_instruction(alloc);
        auto* pwm      = create_pointwise_module(
            p, "main:pointwise0", {x1, x2}, [=](auto* pm, const auto& inputs) {
                auto mul_ins = pm->add_instruction(migraphx::make_op("mul"), inputs[0], inputs[1]);
                return pm->add_instruction(migraphx::make_op("sigmoid"), mul_ins);
            });
        auto pw_op       = migraphx::make_op("pointwise");
        auto pre_comp_op = migraphx::make_op(
            "gpu::precompile_op",
            {{"op", migraphx::to_value(pw_op)}, {"output_shape", migraphx::to_value(s3)}});

        auto pw_ins = mm->add_instruction(pre_comp_op, {x1, x2, alloc_ins}, {pwm});

        auto conv_op = migraphx::make_op("convolution", {{"padding", {1, 1, 1, 1}}});
        auto alloc_ins2 =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(s4)}}));
        auto alloc_ins3 =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(s3)}}));
        auto conv =
            mm->add_instruction(migraphx::make_op("gpu::convolution", {{"op", conv_op.to_value()}}),
                                pw_ins,
                                weights,
                                alloc_ins2,
                                alloc_ins3);
        mm->add_return({conv});
        return p;
    };
    migraphx::program p1 = create_program();
    run_pass(p1);
    migraphx::program p2 = create_fused_program();
    EXPECT(p1 == p2);
}
#endif

TEST_CASE(concat_pointwise_contiguous)
{
    migraphx::shape s1 = migraphx::shape::from_permutation(
        migraphx::shape::float_type, {128, 2, 196, 32}, {0, 2, 1, 3});
    migraphx::shape s2 = migraphx::shape::from_permutation(
        migraphx::shape::float_type, {128, 4, 196, 32}, {0, 2, 1, 3});
    migraphx::shape s3{migraphx::shape::float_type, {128, 4, 196, 32}};
    auto create_program = [=]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x1  = mm->add_parameter("x1", s1);
        auto x2  = mm->add_parameter("x2", s1);
        auto y   = mm->add_parameter("y", s2);

        auto concat_op = migraphx::make_op("concat", {{"axis", 1}});
        auto concat_precompile_op =
            migraphx::make_op("gpu::precompile_op", {{"op", migraphx::to_value(concat_op)}});
        auto x_alloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(s2)}}));
        auto x         = mm->add_instruction(concat_precompile_op, {x1, x2, x_alloc});
        auto alloc     = migraphx::make_op("allocate", {{"shape", to_value(s3)}});
        auto alloc_ins = mm->add_instruction(alloc);
        auto* pw_add1 =
            create_pointwise_module(p, "main:pointwise0", {x, y}, single_pointwise("add"));

        auto pw_op       = migraphx::make_op("pointwise");
        auto pre_comp_op = migraphx::make_op(
            "gpu::precompile_op",
            {{"op", migraphx::to_value(pw_op)}, {"output_shape", migraphx::to_value(s3)}});
        auto add1 = mm->add_instruction(pre_comp_op, {x, y, alloc_ins}, {pw_add1});
        auto rsp =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {25088, 128}}}), add1);
        mm->add_return({rsp});
        return p;
    };
    auto create_fused_program = [=]() {
        migraphx::program p;
        auto* mm                  = p.get_main_module();
        auto x1                   = mm->add_parameter("x1", s1);
        auto x2                   = mm->add_parameter("x2", s1);
        auto y                    = mm->add_parameter("y", s2);
        auto concat_op            = migraphx::make_op("concat", {{"axis", 1}});
        auto concat_precompile_op = migraphx::make_op("gpu::precompile_op",
                                                      {{"op", migraphx::to_value(concat_op)},
                                                       {"additional_args", 2},
                                                       {"ignore_modules", true},
                                                       {"output_shape", migraphx::to_value(s3)}});
        auto alloc                = migraphx::make_op("allocate", {{"shape", to_value(s3)}});
        auto alloc_ins            = mm->add_instruction(alloc);
        // use y's input shape for creating pointwise module for both the params
        auto* pw_add1 =
            create_pointwise_module(p, "main:pointwise0", {y, y}, single_pointwise("add"));
        auto x     = mm->add_instruction(concat_precompile_op, {x1, x2, y, alloc_ins}, {pw_add1});
        auto pw_op = migraphx::make_op("pointwise");
        auto rsp =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {25088, 128}}}), x);
        mm->add_return({rsp});
        return p;
    };
    migraphx::program p1 = create_program();
    run_pass(p1);
    migraphx::program p2 = create_fused_program();
    EXPECT(p1 == p2);
}

TEST_CASE(channelwise_conv_pointwise)
{
    migraphx::shape sx{migraphx::shape::float_type, {2, 4, 8, 8}};
    migraphx::shape sw{migraphx::shape::float_type, {4, 1, 3, 3}};
    migraphx::shape sout{migraphx::shape::float_type, {2, 4, 6, 6}};

    auto create_program = [=](bool first_arg_conv) {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto x         = mm->add_parameter("x", sx);
        auto w         = mm->add_parameter("w", sw);
        auto z         = mm->add_parameter("z", sout);
        auto alloc     = migraphx::make_op("allocate", {{"shape", to_value(sout)}});
        auto alloc_ins = mm->add_instruction(alloc);
        auto conv_ins =
            mm->add_instruction(make_precompile_op("gpu::channelwise_conv"), x, w, alloc_ins);
        std::vector<migraphx::instruction_ref> pw_inputs = {conv_ins, z};
        if(not first_arg_conv)
        {
            pw_inputs = {z, conv_ins};
        }
        auto* pw_add =
            create_pointwise_module(p, "main:pointwise0", pw_inputs, single_pointwise("add"));
        auto alloc_ins2 = mm->add_instruction(alloc);
        pw_inputs.push_back(alloc_ins2);
        auto add_ins = mm->add_instruction(make_precompile_op("pointwise"), pw_inputs, {pw_add});
        mm->add_return({add_ins});
        return p;
    };

    auto create_fused_program = [=]() {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto x         = mm->add_parameter("x", sx);
        auto w         = mm->add_parameter("w", sw);
        auto z         = mm->add_parameter("z", sout);
        auto alloc     = migraphx::make_op("allocate", {{"shape", to_value(sout)}});
        auto alloc_ins = mm->add_instruction(alloc);
        auto* pw_add =
            create_pointwise_module(p, "main:pointwise0", {x, z}, single_pointwise("add"));
        auto conv_op     = migraphx::make_op("gpu::channelwise_conv");
        auto pre_comp_op = migraphx::make_op(
            "gpu::precompile_op",
            {{"op", migraphx::to_value(conv_op)}, {"output_shape", migraphx::to_value(sout)}});
        auto fused_ins = mm->add_instruction(pre_comp_op, {x, w, z, alloc_ins}, {pw_add});
        mm->add_return({fused_ins});
        return p;
    };

    {
        migraphx::program p1 = create_program(true);
        run_pass(p1);
        migraphx::program p2 = create_fused_program();
        EXPECT(p1 == p2);
    }
    {
        // conv is not arg(0), should not fuse
        migraphx::program p1 = create_program(false);
        run_pass(p1);
        EXPECT(p1 == create_program(false));
    }
}

TEST_CASE(channelwise_conv_pointwise_already_fused)
{
    migraphx::shape sx{migraphx::shape::float_type, {2, 4, 8, 8}};
    migraphx::shape sw{migraphx::shape::float_type, {4, 1, 3, 3}};
    migraphx::shape sout{migraphx::shape::float_type, {2, 4, 6, 6}};

    auto create_program = [=]() {
        migraphx::program p;
        auto* mm       = p.get_main_module();
        auto x         = mm->add_parameter("x", sx);
        auto w         = mm->add_parameter("w", sw);
        auto z         = mm->add_parameter("z", sout);
        auto y         = mm->add_parameter("y", sout);
        auto alloc     = migraphx::make_op("allocate", {{"shape", to_value(sout)}});
        auto alloc_ins = mm->add_instruction(alloc);
        // channelwise_conv already has a module (already fused)
        auto* pw_relu =
            create_pointwise_module(p, "main:pointwise0", {x}, [](auto* pm, const auto& inputs) {
                return pm->add_instruction(migraphx::make_op("relu"), inputs[0]);
            });
        auto conv_op     = migraphx::make_op("gpu::channelwise_conv");
        auto pre_comp_op = migraphx::make_op(
            "gpu::precompile_op",
            {{"op", migraphx::to_value(conv_op)}, {"output_shape", migraphx::to_value(sout)}});
        auto conv_ins = mm->add_instruction(pre_comp_op, {x, w, z, alloc_ins}, {pw_relu});
        auto* pw_add =
            create_pointwise_module(p, "main:pointwise1", {conv_ins, y}, single_pointwise("add"));
        auto alloc_ins2 = mm->add_instruction(alloc);
        auto add_ins    = mm->add_instruction(
            make_precompile_op("pointwise"), {conv_ins, y, alloc_ins2}, {pw_add});
        mm->add_return({add_ins});
        return p;
    };

    // Should not fuse since channelwise_conv already has a module
    migraphx::program p1 = create_program();
    run_pass(p1);
    EXPECT(p1 == create_program());
}

TEST_CASE(precompile_copy)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};

    auto create_program = [=]() {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto alloc1  = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(s)}}));
        auto* pw_mod = create_pointwise_module(p, "main:pointwise0", {x}, single_pointwise("relu"));
        auto pw      = mm->add_instruction(make_precompile_op("pointwise"), {x, alloc1}, {pw_mod});
        auto alloc2  = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(s)}}));
        auto copy    = mm->add_instruction(migraphx::make_op("hip::copy"), pw, alloc2);
        mm->add_return({copy});
        return p;
    };

    auto create_fused_program = [=]() {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto alloc   = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(s)}}));
        auto* pw_mod = create_pointwise_module(p, "main:pointwise0", {x}, single_pointwise("relu"));
        auto pw_op   = migraphx::make_op("pointwise");
        auto pre_comp_op = migraphx::make_op(
            "gpu::precompile_op",
            {{"op", migraphx::to_value(pw_op)}, {"output_shape", migraphx::to_value(s)}});
        auto pw = mm->add_instruction(pre_comp_op, {x, alloc}, {pw_mod});
        mm->add_return({pw});
        return p;
    };

    migraphx::program p1 = create_program();
    run_pass(p1);
    migraphx::program p2 = create_fused_program();
    EXPECT(p1 == p2);
}

TEST_CASE(precompile_copy_non_alloc)
{
    // After eliminate_concat the precompile_op's output buffer may be a slice of
    // a shared super-allocation rather than a plain allocate.  Merging a second
    // hip::copy on top would redirect the write away from the slice and leave
    // the first destination unwritten.  Verify the merge is blocked.
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape super_s{migraphx::shape::float_type, {2, 3, 8}};
    migraphx::shape slice_s{migraphx::shape::float_type, {2, 3, 4}, {24, 8, 1}};

    auto create_program = [=]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto super_alloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(super_s)}}));
        auto slice = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {4}}}),
            super_alloc);
        auto* pw_mod = create_pointwise_module(p, "main:pointwise0", {x}, single_pointwise("relu"));
        auto pw_op   = migraphx::make_op("pointwise");
        auto pre_comp_op = migraphx::make_op(
            "gpu::precompile_op",
            {{"op", migraphx::to_value(pw_op)}, {"output_shape", migraphx::to_value(slice_s)}});
        auto pw   = mm->add_instruction(pre_comp_op, {x, slice}, {pw_mod});
        auto dest = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(s)}}));
        auto copy = mm->add_instruction(migraphx::make_op("hip::copy"), pw, dest);
        mm->add_return({copy});
        return p;
    };

    // hip::copy is NOT merged into the precompile_op and is left as-is (it is lowered later by
    // lower_device_ops, not fuse_ops).
    auto create_expected_program = [=]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto super_alloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(super_s)}}));
        auto slice = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {4}}}),
            super_alloc);
        auto* pw_mod = create_pointwise_module(p, "main:pointwise0", {x}, single_pointwise("relu"));
        auto pw_op   = migraphx::make_op("pointwise");
        auto pre_comp_op = migraphx::make_op(
            "gpu::precompile_op",
            {{"op", migraphx::to_value(pw_op)}, {"output_shape", migraphx::to_value(slice_s)}});
        auto pw   = mm->add_instruction(pre_comp_op, {x, slice}, {pw_mod});
        auto dest = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(s)}}));
        auto copy = mm->add_instruction(migraphx::make_op("hip::copy"), pw, dest);
        mm->add_return({copy});
        return p;
    };

    migraphx::program p1 = create_program();
    run_pass(p1);
    migraphx::program p2 = create_expected_program();
    EXPECT(p1 == p2);
}

TEST_CASE(precompile_copy_chained)
{
    // Simulate the nested concat scenario: a chain of hip::copy operations
    // where the precompile_op feeds into two copies through eliminate_concat.
    // Only the first copy should be merged; the second must remain.
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::shape super_s1{migraphx::shape::float_type, {2, 3, 8}};
    migraphx::shape super_s2{migraphx::shape::float_type, {2, 3, 8}};
    migraphx::shape slice_s{migraphx::shape::float_type, {2, 3, 4}, {24, 8, 1}};

    auto create_program = [=]() {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto alloc   = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(s)}}));
        auto* pw_mod = create_pointwise_module(p, "main:pointwise0", {x}, single_pointwise("relu"));
        // precompile_op with a plain allocation — eligible for first merge
        auto pw = mm->add_instruction(make_precompile_op("pointwise"), {x, alloc}, {pw_mod});
        // First copy into slice of super-alloc 1 (from eliminate_concat for concat1)
        auto super1 =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(super_s1)}}));
        auto slice1 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {4}}}), super1);
        auto copy1 = mm->add_instruction(migraphx::make_op("hip::copy"), pw, slice1);
        // Second copy into slice of super-alloc 2 (from eliminate_concat for concat2)
        auto super2 =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(super_s2)}}));
        auto slice2 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {4}}}), super2);
        auto copy2 = mm->add_instruction(migraphx::make_op("hip::copy"), copy1, slice2);
        mm->add_return({copy2});
        return p;
    };

    auto create_expected_program = [=]() {
        migraphx::program p;
        auto* mm     = p.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto* pw_mod = create_pointwise_module(p, "main:pointwise0", {x}, single_pointwise("relu"));
        // First merge happened: precompile_op now writes to slice1
        auto super1 =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(super_s1)}}));
        auto slice1 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {4}}}), super1);
        auto pw_op       = migraphx::make_op("pointwise");
        auto pre_comp_op = migraphx::make_op(
            "gpu::precompile_op",
            {{"op", migraphx::to_value(pw_op)}, {"output_shape", migraphx::to_value(slice_s)}});
        auto pw = mm->add_instruction(pre_comp_op, {x, slice1}, {pw_mod});
        // Second copy is NOT merged and is left as-is (lowered later by lower_device_ops).
        auto super2 =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(super_s2)}}));
        auto slice2 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {4}}}), super2);
        auto copy2 = mm->add_instruction(migraphx::make_op("hip::copy"), pw, slice2);
        mm->add_return({copy2});
        return p;
    };

    migraphx::program p1 = create_program();
    run_pass(p1);
    migraphx::program p2 = create_expected_program();
    EXPECT(p1 == p2);
}

TEST_CASE(concat_pointwise_multi_use)
{
    migraphx::shape s1 = migraphx::shape::from_permutation(
        migraphx::shape::float_type, {128, 2, 196, 32}, {0, 2, 1, 3});
    migraphx::shape s2 = migraphx::shape::from_permutation(
        migraphx::shape::float_type, {128, 4, 196, 32}, {0, 2, 1, 3});
    migraphx::shape s3{migraphx::shape::float_type, {128, 4, 196, 32}};
    auto create_program = [=]() {
        migraphx::program p;
        auto* mm = p.get_main_module();
        auto x1  = mm->add_parameter("x1", s1);
        auto x2  = mm->add_parameter("x2", s1);
        auto y   = mm->add_parameter("y", s2);

        auto concat_op = migraphx::make_op("concat", {{"axis", 1}});
        auto concat_precompile_op =
            migraphx::make_op("gpu::precompile_op", {{"op", migraphx::to_value(concat_op)}});
        auto x_alloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(s2)}}));
        auto x         = mm->add_instruction(concat_precompile_op, {x1, x2, x_alloc});
        auto alloc     = migraphx::make_op("allocate", {{"shape", to_value(s3)}});
        auto alloc_ins = mm->add_instruction(alloc);
        auto* pw_add1 =
            create_pointwise_module(p, "main:pointwise0", {x, y}, single_pointwise("add"));

        auto pw_op       = migraphx::make_op("pointwise");
        auto pre_comp_op = migraphx::make_op(
            "gpu::precompile_op",
            {{"op", migraphx::to_value(pw_op)}, {"output_shape", migraphx::to_value(s3)}});
        auto add1 = mm->add_instruction(pre_comp_op, {x, y, alloc_ins}, {pw_add1});

        // Second use of concat result
        auto alloc_ins2 = mm->add_instruction(alloc);
        auto* pw_add2 =
            create_pointwise_module(p, "main:pointwise1", {x, y}, single_pointwise("mul"));
        auto pre_comp_op2 = migraphx::make_op(
            "gpu::precompile_op",
            {{"op", migraphx::to_value(pw_op)}, {"output_shape", migraphx::to_value(s3)}});
        auto mul1 = mm->add_instruction(pre_comp_op2, {x, y, alloc_ins2}, {pw_add2});

        mm->add_return({add1, mul1});
        return p;
    };
    migraphx::program p1 = create_program();
    migraphx::program p2 = create_program();
    run_pass(p1);
    EXPECT(p1 == p2);
}

// A table that is a view of a stored tensor is read through the view, so
// there is nothing to fuse
TEST_CASE(gather_dot_weights_view)
{
    migraphx::shape out{migraphx::shape::float_type, {8, 5}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 5, 3}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto wt =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), w);
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto* im   = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", {wt->get_shape().type(), wt->get_shape().lens()})});
        auto dot = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, wt, alloc}, {im});
        mm->add_return({dot});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

// The pointwise operator computing the table becomes the weights submodule:
// the transpose of its result is applied to its input instead, and the bias
// stays after the weight tensors
TEST_CASE(gather_dot_weights_pointwise)
{
    migraphx::shape ws{migraphx::shape::float_type, {4, 5, 3}};
    migraphx::shape out{migraphx::shape::float_type, {8, 5}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w      = mm->add_parameter("w", ws);
        auto idx    = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto b      = mm->add_parameter("b", {migraphx::shape::float_type, {4, 1, 5}});
        auto* pm    = create_pointwise_module(p1, "main:pointwise0", {w}, single_pointwise("relu"));
        auto walloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(ws)}}));
        auto wr     = mm->add_instruction(make_precompile_op("pointwise"), {w, walloc}, {pm});
        auto wt =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), wr);
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto* im   = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", {wt->get_shape().type(), wt->get_shape().lens()})});
        auto dot = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, wt, b, alloc}, {im});
        mm->add_return({dot});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w   = mm->add_parameter("w", ws);
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto b   = mm->add_parameter("b", {migraphx::shape::float_type, {4, 1, 5}});
        auto* pm = create_pointwise_module(p2, "main:pointwise0", {w}, single_pointwise("relu"));
        auto wt =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), w);
        auto* wm = p2.create_module("main:pointwise0:gather_dot");
        wm->set_bypass();
        auto x0 = wm->add_parameter("x0", {migraphx::shape::float_type, {4, 3, 5}});
        wm->add_return({wm->add_instruction(migraphx::make_op("pointwise"), {x0}, {pm})});
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto dot   = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, wt, b, alloc}, {wm});
        mm->add_return({dot});
    }
    EXPECT(p1.sort() == p2.sort());
}

// The bytes of the unpack_int4 the pointwise operator reads are read as they
// are stored: the weights submodule unpacks them along the axis the transpose
// moves the unpacked axis to
TEST_CASE(gather_dot_weights_pointwise_unpack_int4)
{
    migraphx::shape out{migraphx::shape::half_type, {8, 6}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", {migraphx::shape::half_type, {8, 4}});
        auto w      = mm->add_parameter("w", {migraphx::shape::uint8_type, {4, 6, 2}});
        auto scales = mm->add_parameter("scales", {migraphx::shape::half_type, {4, 6}});
        auto idx    = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto ualloc = mm->add_instruction(migraphx::make_op(
            "allocate",
            {{"shape", to_value(migraphx::shape{migraphx::shape::uint8_type, {4, 6, 4}})}}));
        auto wu     = mm->add_instruction(
            make_precompile_op(migraphx::make_op("unpack_int4", {{"axis", 2}})), w, ualloc);
        auto su = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), scales);
        auto sb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 6, 4}}}), su);
        auto* pm = create_pointwise_module(
            p1, "main:pointwise0", {wu, sb}, [](auto* m, const auto& inputs) {
                auto q = m->add_instruction(
                    migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
                    inputs[0]);
                return m->add_instruction(migraphx::make_op("mul"), q, inputs[1]);
            });
        auto palloc = mm->add_instruction(migraphx::make_op(
            "allocate",
            {{"shape", to_value(migraphx::shape{migraphx::shape::half_type, {4, 6, 4}})}}));
        auto dq     = mm->add_instruction(make_precompile_op("pointwise"), {wu, sb, palloc}, {pm});
        auto wt =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), dq);
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto* im   = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", {wt->get_shape().type(), wt->get_shape().lens()})});
        auto dot = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, wt, alloc}, {im});
        mm->add_return({dot});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm    = p2.get_main_module();
        auto x      = mm->add_parameter("x", {migraphx::shape::half_type, {8, 4}});
        auto w      = mm->add_parameter("w", {migraphx::shape::uint8_type, {4, 6, 2}});
        auto scales = mm->add_parameter("scales", {migraphx::shape::half_type, {4, 6}});
        auto idx    = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto su     = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {2}}}), scales);
        auto sb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 6, 4}}}), su);
        auto* pm = create_pointwise_module(
            p2, "main:pointwise0", {w, sb}, [](auto* m, const auto& inputs) {
                auto q = m->add_instruction(
                    migraphx::make_op("convert", {{"target_type", migraphx::shape::half_type}}),
                    inputs[0]);
                return m->add_instruction(migraphx::make_op("mul"), q, inputs[1]);
            });
        auto wt =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), w);
        auto st =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), sb);
        auto* wm = p2.create_module("main:pointwise0:gather_dot");
        wm->set_bypass();
        auto x0 = wm->add_parameter("x0", {migraphx::shape::uint8_type, {4, 2, 6}});
        auto x1 = wm->add_parameter("x1", {migraphx::shape::half_type, {4, 4, 6}});
        auto u  = wm->add_instruction(migraphx::make_op("unpack_int4", {{"axis", 1}}), x0);
        wm->add_return({wm->add_instruction(migraphx::make_op("pointwise"), {u, x1}, {pm})});
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto dot   = mm->add_instruction(make_precompile_op(migraphx::make_op("gpu::gather_dot")),
                                         {x, idx, wt, st, alloc},
                                         {wm});
        mm->add_return({dot});
    }
    EXPECT(p1.sort() == p2.sort());
}

// A pointwise operator in more axes than the table, as with a scale for each
// block of k: its inputs and the rows are viewed in the common dimensions,
// where the reshape that merges the block axes is not needed
TEST_CASE(gather_dot_weights_pointwise_common_dims)
{
    migraphx::shape ws{migraphx::shape::float_type, {4, 5, 2, 3}};
    migraphx::shape out{migraphx::shape::float_type, {8, 5}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 6}});
        auto w   = mm->add_parameter("w", ws);
        auto s   = mm->add_parameter("s", {migraphx::shape::float_type, {4, 5, 2}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto sb  = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {4, 5, 2, 3}}}), s);
        auto* pm = create_pointwise_module(p1, "main:pointwise0", {w, sb}, single_pointwise("mul"));
        auto walloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(ws)}}));
        auto ws4    = mm->add_instruction(make_precompile_op("pointwise"), {w, sb, walloc}, {pm});
        auto wt     = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), ws4);
        auto table =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {4, 6, 5}}}), wt);
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto* im   = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return(
            {im->add_parameter("x0", {table->get_shape().type(), table->get_shape().lens()})});
        auto dot = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, table, alloc}, {im});
        mm->add_return({dot});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 6}});
        auto w   = mm->add_parameter("w", ws);
        auto s   = mm->add_parameter("s", {migraphx::shape::float_type, {4, 5, 2}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto sb  = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {4, 5, 2, 3}}}), s);
        auto* pm = create_pointwise_module(p2, "main:pointwise0", {w, sb}, single_pointwise("mul"));
        auto wt =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), w);
        auto st = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 3, 1}}}), sb);
        auto* wm = p2.create_module("main:pointwise0:gather_dot");
        wm->set_bypass();
        auto x0 = wm->add_parameter("x0", {migraphx::shape::float_type, {4, 2, 3, 5}});
        auto x1 = wm->add_parameter("x1", {migraphx::shape::float_type, {4, 2, 3, 5}});
        wm->add_return({wm->add_instruction(migraphx::make_op("pointwise"), {x0, x1}, {pm})});
        auto rows =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {8, 2, 3}}}), x);
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto dot   = mm->add_instruction(make_precompile_op(migraphx::make_op("gpu::gather_dot")),
                                         {rows, idx, wt, st, alloc},
                                         {wm});
        mm->add_return({dot});
    }
    EXPECT(p1.sort() == p2.sort());
}

// A reshape that merges axes of the columns: the sums are computed in the
// common dimensions and reshaped after
TEST_CASE(gather_dot_weights_pointwise_split_columns)
{
    migraphx::shape ws{migraphx::shape::float_type, {4, 3, 5, 2}};
    migraphx::shape out{migraphx::shape::float_type, {8, 10}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w      = mm->add_parameter("w", ws);
        auto idx    = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto* pm    = create_pointwise_module(p1, "main:pointwise0", {w}, single_pointwise("relu"));
        auto walloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(ws)}}));
        auto wr     = mm->add_instruction(make_precompile_op("pointwise"), {w, walloc}, {pm});
        auto table =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {4, 3, 10}}}), wr);
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto* im   = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return(
            {im->add_parameter("x0", {table->get_shape().type(), table->get_shape().lens()})});
        auto dot = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, table, alloc}, {im});
        mm->add_return({dot});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w   = mm->add_parameter("w", ws);
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto* pm = create_pointwise_module(p2, "main:pointwise0", {w}, single_pointwise("relu"));
        auto* wm = p2.create_module("main:pointwise0:gather_dot");
        wm->set_bypass();
        auto x0 = wm->add_parameter("x0", ws);
        wm->add_return({wm->add_instruction(migraphx::make_op("pointwise"), {x0}, {pm})});
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto alloc_view =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {8, 5, 2}}}), alloc);
        auto sums = mm->add_instruction(make_precompile_op(migraphx::make_op("gpu::gather_dot")),
                                        {x, idx, w, alloc_view},
                                        {wm});
        auto dot =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {8, 10}}}), sums);
        mm->add_return({dot});
    }
    EXPECT(p1.sort() == p2.sort());
}

// A reshape that merges the entries with another axis: the indices select
// along one axis, so the pointwise operator is not fused
TEST_CASE(gather_dot_weights_pointwise_merged_entries)
{
    migraphx::shape ws{migraphx::shape::float_type, {2, 2, 3, 5}};
    migraphx::shape out{migraphx::shape::float_type, {8, 5}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w      = mm->add_parameter("w", ws);
        auto idx    = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto* pm    = create_pointwise_module(p1, "main:pointwise0", {w}, single_pointwise("relu"));
        auto walloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(ws)}}));
        auto wr     = mm->add_instruction(make_precompile_op("pointwise"), {w, walloc}, {pm});
        auto table =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {4, 3, 5}}}), wr);
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto* im   = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return(
            {im->add_parameter("x0", {table->get_shape().type(), table->get_shape().lens()})});
        auto dot = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, table, alloc}, {im});
        mm->add_return({dot});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

// A pointwise operator with another reader computes the table for it too, so
// it is not fused
TEST_CASE(gather_dot_weights_pointwise_shared)
{
    migraphx::shape ws{migraphx::shape::float_type, {4, 5, 3}};
    migraphx::shape out{migraphx::shape::float_type, {8, 5}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w      = mm->add_parameter("w", ws);
        auto idx    = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto* pm    = create_pointwise_module(p1, "main:pointwise0", {w}, single_pointwise("relu"));
        auto walloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(ws)}}));
        auto wr     = mm->add_instruction(make_precompile_op("pointwise"), {w, walloc}, {pm});
        auto wt =
            mm->add_instruction(migraphx::make_op("transpose", {{"permutation", {0, 2, 1}}}), wr);
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto* im   = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", {wt->get_shape().type(), wt->get_shape().lens()})});
        auto dot = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, wt, alloc}, {im});
        mm->add_return({dot, wr});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

// A table that is read as it is stored has nothing to fuse
TEST_CASE(gather_dot_weights_stored)
{
    migraphx::shape out{migraphx::shape::float_type, {8, 5}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w     = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx   = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto* im   = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", w->get_shape())});
        auto dot = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, w, alloc}, {im});
        mm->add_return({dot});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

// The pointwise operator reading the sums, here an add of another input,
// becomes the output submodule over groups of one column
TEST_CASE(gather_dot_pointwise)
{
    migraphx::shape out{migraphx::shape::float_type, {8, 5}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto b   = mm->add_parameter("b", out);
        auto dalloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto* im = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", w->get_shape())});
        auto dot = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, w, dalloc}, {im});
        auto* pm =
            create_pointwise_module(p1, "main:pointwise0", {b, dot}, single_pointwise("add"));
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto add   = mm->add_instruction(make_precompile_op("pointwise"), {b, dot, alloc}, {pm});
        mm->add_return({add});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto b   = mm->add_parameter("b", out);
        auto* pm = create_pointwise_module(p2, "main:pointwise0", {b, b}, single_pointwise("add"));
        auto* im = p2.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", w->get_shape())});
        auto* om = p2.create_module("main:pointwise0:gather_dot");
        om->set_bypass();
        auto sums  = om->add_parameter("x0", out);
        auto other = om->add_parameter("x1", out);
        om->add_return({om->add_instruction(migraphx::make_op("pointwise"), {other, sums}, {pm})});
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto dot   = mm->add_instruction(make_precompile_op(migraphx::make_op("gpu::gather_dot")),
                                         {x, idx, w, b, alloc},
                                         {im, om});
        mm->add_return({dot});
    }
    EXPECT(p1.sort() == p2.sort());
}

// The other inputs of the output submodule follow the bias of the gather_dot
TEST_CASE(gather_dot_pointwise_bias)
{
    migraphx::shape out{migraphx::shape::float_type, {8, 5}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w    = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx  = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto bias = mm->add_parameter("bias", {migraphx::shape::float_type, {4, 1, 5}});
        auto y    = mm->add_parameter("y", out);
        auto dalloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto* im = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", w->get_shape())});
        auto dot = mm->add_instruction(make_precompile_op(migraphx::make_op("gpu::gather_dot")),
                                       {x, idx, w, bias, dalloc},
                                       {im});
        auto* pm =
            create_pointwise_module(p1, "main:pointwise0", {dot, y}, single_pointwise("mul"));
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto mul   = mm->add_instruction(make_precompile_op("pointwise"), {dot, y, alloc}, {pm});
        mm->add_return({mul});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w    = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx  = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto bias = mm->add_parameter("bias", {migraphx::shape::float_type, {4, 1, 5}});
        auto y    = mm->add_parameter("y", out);
        auto* pm  = create_pointwise_module(p2, "main:pointwise0", {y, y}, single_pointwise("mul"));
        auto* im  = p2.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", w->get_shape())});
        auto* om = p2.create_module("main:pointwise0:gather_dot");
        om->set_bypass();
        auto sums  = om->add_parameter("x0", out);
        auto other = om->add_parameter("x1", out);
        om->add_return({om->add_instruction(migraphx::make_op("pointwise"), {sums, other}, {pm})});
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto dot   = mm->add_instruction(make_precompile_op(migraphx::make_op("gpu::gather_dot")),
                                         {x, idx, w, bias, y, alloc},
                                         {im, om});
        mm->add_return({dot});
    }
    EXPECT(p1.sort() == p2.sort());
}

// The sums have another reader, so they are still computed on their own
TEST_CASE(gather_dot_pointwise_multi_use)
{
    migraphx::shape out{migraphx::shape::float_type, {8, 5}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto b   = mm->add_parameter("b", out);
        auto dalloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto* im = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", w->get_shape())});
        auto dot = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, w, dalloc}, {im});
        auto* pm =
            create_pointwise_module(p1, "main:pointwise0", {b, dot}, single_pointwise("add"));
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto add   = mm->add_instruction(make_precompile_op("pointwise"), {b, dot, alloc}, {pm});
        mm->add_return({add, dot});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

// The pointwise operator reads the sums through a view, which the output
// submodule holds. Its other input is read by row.
TEST_CASE(gather_dot_pointwise_view)
{
    migraphx::shape sums{migraphx::shape::float_type, {8, 5}};
    migraphx::shape gathered{migraphx::shape::float_type, {8, 5}};
    migraphx::shape out{migraphx::shape::float_type, {8, 1, 5}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto b   = mm->add_parameter("b", {migraphx::shape::float_type, {4, 5}});
        auto galloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(gathered)}}));
        auto bg = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gather", {{"axis", 0}})), b, idx, galloc);
        auto bu = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), bg);
        auto dalloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(sums)}}));
        auto* im = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", w->get_shape())});
        auto dot = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, w, dalloc}, {im});
        auto du = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), dot);
        auto* pm =
            create_pointwise_module(p1, "main:pointwise0", {bu, du}, single_pointwise("add"));
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto add   = mm->add_instruction(make_precompile_op("pointwise"), {bu, du, alloc}, {pm});
        mm->add_return({add});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 5}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto b   = mm->add_parameter("b", {migraphx::shape::float_type, {4, 5}});
        auto galloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(gathered)}}));
        auto bg = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gather", {{"axis", 0}})), b, idx, galloc);
        auto bu  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), bg);
        auto* pm =
            create_pointwise_module(p2, "main:pointwise0", {bu, bu}, single_pointwise("add"));
        auto* im = p2.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", w->get_shape())});
        auto* om = p2.create_module("main:pointwise0:gather_dot");
        om->set_bypass();
        auto x0   = om->add_parameter("x0", sums);
        auto x1   = om->add_parameter("x1", out);
        auto view = om->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x0);
        om->add_return({om->add_instruction(migraphx::make_op("pointwise"), {x1, view}, {pm})});
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto dot   = mm->add_instruction(make_precompile_op(migraphx::make_op("gpu::gather_dot")),
                                         {x, idx, w, bu, alloc},
                                         {im, om});
        mm->add_return({dot});
    }
    EXPECT(p1.sort() == p2.sort());
}

// The pointwise operator reading both columns of every pair of the sums
// becomes the output submodule, which gives one column per pair
TEST_CASE(gather_dot_group)
{
    migraphx::shape sums{migraphx::shape::float_type, {8, 6}};
    migraphx::shape out{migraphx::shape::float_type, {8, 3, 1}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 6}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto dalloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(sums)}}));
        auto* im = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", w->get_shape())});
        auto dot = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, w, dalloc}, {im});
        auto pairs =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {8, 3, 2}}}), dot);
        auto first = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {1}}}), pairs);
        auto second = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {2}}}), pairs);
        auto* pm = create_pointwise_module(
            p1, "main:pointwise0", {second, first}, single_pointwise("sub"));
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto sub =
            mm->add_instruction(make_precompile_op("pointwise"), {second, first, alloc}, {pm});
        mm->add_return({sub});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 6}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto* pm = create_pointwise_module(p2, "main:pointwise0", {x, x}, single_pointwise("sub"));
        auto* im = p2.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", w->get_shape())});
        auto* om = p2.create_module("main:pointwise0:gather_dot");
        om->set_bypass();
        auto x0 = om->add_parameter("x0", sums);
        auto pairs =
            om->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {8, 3, 2}}}), x0);
        auto second = om->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {2}}}), pairs);
        auto first = om->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {1}}}), pairs);
        om->add_return(
            {om->add_instruction(migraphx::make_op("pointwise"), {second, first}, {pm})});
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto dot   = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, w, alloc}, {im, om});
        mm->add_return({dot});
    }
    EXPECT(p1.sort() == p2.sort());
}

// An add of another input then a pointwise operator over the pairs: there is
// one output submodule, so only the add is fused
TEST_CASE(gather_dot_pointwise_then_group)
{
    migraphx::shape sums{migraphx::shape::float_type, {8, 6}};
    migraphx::shape out{migraphx::shape::float_type, {8, 3, 1}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 6}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto b   = mm->add_parameter("b", sums);
        auto y   = mm->add_parameter("y", out);
        auto dalloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(sums)}}));
        auto* im = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", w->get_shape())});
        auto dot = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, w, dalloc}, {im});
        auto* pm0 =
            create_pointwise_module(p1, "main:pointwise0", {b, dot}, single_pointwise("add"));
        auto aalloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(sums)}}));
        auto add = mm->add_instruction(make_precompile_op("pointwise"), {b, dot, aalloc}, {pm0});
        auto pairs =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {8, 3, 2}}}), add);
        auto first = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {1}}}), pairs);
        auto second = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {2}}}), pairs);
        auto* pm1 = create_pointwise_module(
            p1, "main:pointwise1", {first, y, second}, [](auto* m, const auto& inputs) {
                auto mul = m->add_instruction(migraphx::make_op("mul"), inputs[0], inputs[1]);
                return m->add_instruction(migraphx::make_op("sub"), mul, inputs[2]);
            });
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto pw =
            mm->add_instruction(make_precompile_op("pointwise"), {first, y, second, alloc}, {pm1});
        mm->add_return({pw});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w    = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 6}});
        auto idx  = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto b    = mm->add_parameter("b", sums);
        auto y    = mm->add_parameter("y", out);
        auto* pm0 = create_pointwise_module(p2, "main:pointwise0", {b, b}, single_pointwise("add"));
        auto aalloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(sums)}}));
        auto* im = p2.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", w->get_shape())});
        auto* om = p2.create_module("main:pointwise0:gather_dot");
        om->set_bypass();
        auto x0 = om->add_parameter("x0", sums);
        auto x1 = om->add_parameter("x1", sums);
        om->add_return({om->add_instruction(migraphx::make_op("pointwise"), {x1, x0}, {pm0})});
        auto add = mm->add_instruction(make_precompile_op(migraphx::make_op("gpu::gather_dot")),
                                       {x, idx, w, b, aalloc},
                                       {im, om});
        auto pairs =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {8, 3, 2}}}), add);
        auto first = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {1}}}), pairs);
        auto second = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {2}}}), pairs);
        auto* pm1 = create_pointwise_module(
            p2, "main:pointwise1", {first, y, second}, [](auto* m, const auto& inputs) {
                auto mul = m->add_instruction(migraphx::make_op("mul"), inputs[0], inputs[1]);
                return m->add_instruction(migraphx::make_op("sub"), mul, inputs[2]);
            });
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto pw =
            mm->add_instruction(make_precompile_op("pointwise"), {first, y, second, alloc}, {pm1});
        mm->add_return({pw});
    }
    EXPECT(p1.sort() == p2.sort());
}

// A pointwise operator over the pairs with another input, which follows the
// inputs of the gather_dot
TEST_CASE(gather_dot_group_other_input)
{
    migraphx::shape sums{migraphx::shape::float_type, {8, 6}};
    migraphx::shape out{migraphx::shape::float_type, {8, 3, 1}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 6}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto y   = mm->add_parameter("y", out);
        auto dalloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(sums)}}));
        auto* im = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", w->get_shape())});
        auto dot = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, w, dalloc}, {im});
        auto pairs =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {8, 3, 2}}}), dot);
        auto first = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {1}}}), pairs);
        auto second = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {2}}}), pairs);
        auto* pm = create_pointwise_module(
            p1, "main:pointwise0", {first, y, second}, [](auto* m, const auto& inputs) {
                auto mul = m->add_instruction(migraphx::make_op("mul"), inputs[0], inputs[1]);
                return m->add_instruction(migraphx::make_op("sub"), mul, inputs[2]);
            });
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto pw =
            mm->add_instruction(make_precompile_op("pointwise"), {first, y, second, alloc}, {pm});
        mm->add_return({pw});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 6}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto y   = mm->add_parameter("y", out);
        auto* pm = create_pointwise_module(
            p2, "main:pointwise0", {y, y, y}, [](auto* m, const auto& inputs) {
                auto mul = m->add_instruction(migraphx::make_op("mul"), inputs[0], inputs[1]);
                return m->add_instruction(migraphx::make_op("sub"), mul, inputs[2]);
            });
        auto* im = p2.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", w->get_shape())});
        auto* om = p2.create_module("main:pointwise0:gather_dot");
        om->set_bypass();
        auto x0 = om->add_parameter("x0", sums);
        auto pairs =
            om->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {8, 3, 2}}}), x0);
        auto first = om->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {1}}}), pairs);
        auto x1     = om->add_parameter("x1", out);
        auto second = om->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {2}}}), pairs);
        om->add_return(
            {om->add_instruction(migraphx::make_op("pointwise"), {first, x1, second}, {pm})});
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto dot   = mm->add_instruction(make_precompile_op(migraphx::make_op("gpu::gather_dot")),
                                         {x, idx, w, y, alloc},
                                         {im, om});
        mm->add_return({dot});
    }
    EXPECT(p1.sort() == p2.sort());
}

// The kernel reads the columns of a group from adjacent lanes, which needs a
// power of two of them: groups of 3 columns stay
TEST_CASE(gather_dot_group_of_three)
{
    migraphx::shape sums{migraphx::shape::float_type, {8, 6}};
    migraphx::shape out{migraphx::shape::float_type, {8, 2, 1}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 6}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto dalloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(sums)}}));
        auto* im = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", w->get_shape())});
        auto dot = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, w, dalloc}, {im});
        auto triples =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {8, 2, 3}}}), dot);
        auto first = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {1}}}), triples);
        auto last = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {2}}, {"ends", {3}}}), triples);
        auto* pm =
            create_pointwise_module(p1, "main:pointwise0", {first, last}, single_pointwise("sub"));
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto sub = mm->add_instruction(make_precompile_op("pointwise"), {first, last, alloc}, {pm});
        mm->add_return({sub});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

// The pairs have another reader, so the sums are still computed on their own
TEST_CASE(gather_dot_group_multi_use)
{
    migraphx::shape sums{migraphx::shape::float_type, {8, 6}};
    migraphx::shape out{migraphx::shape::float_type, {8, 3, 1}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", {migraphx::shape::float_type, {8, 3}});
        auto w   = mm->add_parameter("w", {migraphx::shape::float_type, {4, 3, 6}});
        auto idx = mm->add_parameter("idx", {migraphx::shape::int32_type, {8}});
        auto dalloc =
            mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(sums)}}));
        auto* im = p1.create_module("main:gather_dot0");
        im->set_bypass();
        im->add_return({im->add_parameter("x0", w->get_shape())});
        auto dot = mm->add_instruction(
            make_precompile_op(migraphx::make_op("gpu::gather_dot")), {x, idx, w, dalloc}, {im});
        auto pairs =
            mm->add_instruction(migraphx::make_op("reshape_lazy", {{"dims", {8, 3, 2}}}), dot);
        auto first = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {0}}, {"ends", {1}}}), pairs);
        auto second = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {2}}, {"starts", {1}}, {"ends", {2}}}), pairs);
        auto* pm = create_pointwise_module(
            p1, "main:pointwise0", {second, first}, single_pointwise("sub"));
        auto alloc = mm->add_instruction(migraphx::make_op("allocate", {{"shape", to_value(out)}}));
        auto sub =
            mm->add_instruction(make_precompile_op("pointwise"), {second, first, alloc}, {pm});
        mm->add_return({sub, pairs});
    }
    migraphx::program p2 = p1;
    run_pass(p1);
    EXPECT(p1 == p2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
