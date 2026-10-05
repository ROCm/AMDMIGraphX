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
#include <migraphx/fuse_reduce.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>
#include <reduce.hpp>
#include <pointwise.hpp>

static void run_pass(migraphx::program& p, migraphx::fuse_reduce pass = {})
{
    migraphx::run_passes(p, {pass, migraphx::dead_code_elimination{}});
}

TEST_CASE(single)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), y);
        mm->add_return({rsum1, rsum2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto rsum1 = add_reduce(p2, "main:reduce_sum0", {x}, {1}, single_reduce("reduce_sum"));
        auto rsum2 = add_reduce(p2, "main:reduce_sum1", {y}, {1}, single_reduce("reduce_sum"));
        mm->add_return({rsum1, rsum2});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(single_dyn)
{
    migraphx::shape s{migraphx::shape::float_type, {{1, 3}, {4, 8}}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), y);
        mm->add_return({rsum1, rsum2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto rsum1 = add_reduce(p2, "main:reduce_sum0", {x}, {1}, single_reduce("reduce_sum"));
        auto rsum2 = add_reduce(p2, "main:reduce_sum1", {y}, {1}, single_reduce("reduce_sum"));
        mm->add_return({rsum1, rsum2});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_reduce)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add  = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), add);
        mm->add_return({rsum});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto rsum = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum0",
            {x, y},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto add =
                    add_pointwise(p2, rm, "main:pointwise0", inputs, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_reduce_unfusable_broadcast)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 1, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add  = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto addb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), add);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), addb);
        mm->add_return({rsum});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add  = add_pointwise(p2, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto addb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), add);
        auto rsum =
            add_reduce(p2,
                       "main:reduce_sum0",
                       {addb},
                       {2},
                       [&](auto* rm, const auto& inputs, const auto& axes) {
                           return rm->add_instruction(
                               migraphx::make_op("reduce_sum", {{"axes", axes}}), inputs[0]);
                       });
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_multibroadcast_reduce)
{
    // Same as pointwise_reduce_unfusable_broadcast, but rewriting the broadcast
    // onto the pointwise inputs allows the fusion.
    migraphx::shape s{migraphx::shape::float_type, {2, 1, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto add  = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto addb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), add);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), addb);
        mm->add_return({rsum});
    }
    run_pass(p1, {.enable_rewrite_broadcasts = true});

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto bx =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), x);
        auto by =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), y);
        auto rsum = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum0",
            {bx, by},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto add =
                    add_pointwise(p2, rm, "main:pointwise0", inputs, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(pointwise_broadcast_reduce)
{
    // A rank-changing broadcast between the pointwise and the reduce can only be
    // fused by rewriting the broadcast onto the pointwise inputs.
    migraphx::shape s1{migraphx::shape::float_type, {3}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s1);
        auto y    = mm->add_parameter("y", s1);
        auto add  = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto badd = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", s2.lens()}}), add);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), badd);
        mm->add_return({rsum});
    }
    run_pass(p1, {.enable_rewrite_broadcasts = true});

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s1);
        auto bx  = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", s2.lens()}}), x);
        auto by = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", s2.lens()}}), y);
        auto rsum = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum0",
            {bx, by},
            {0},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto add =
                    add_pointwise(p2, rm, "main:pointwise0", inputs, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_broadcast_reduce_same_axes)
{
    // The broadcast expands both the reduced axis and an axis that was already
    // size 1; the size-1 axis part is hoisted above the reduce so both reduces fuse.
    migraphx::shape s1{migraphx::shape::float_type, {2, 1, 4}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s1);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4}}}), rsum1);
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), rsumb);
        mm->add_return({rsum2});
    }
    run_pass(p1, {.enable_rewrite_broadcasts = true});

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto bx =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4}}}), x);
        auto rsum = add_reduce(
            p2,
            "main:reduce_sum1:main:reduce_sum0_broadcast",
            {bx},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4}}}), rsum1);
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                           rsumb);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_broadcast_pointwise_reduce)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 1, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 4}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s1);
        auto y     = mm->add_parameter("y", s2);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), rsum1);
        auto mul   = add_pointwise(p1, "main:pointwise0", {rsumb, y}, single_pointwise("mul"));
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), mul);
        mm->add_return({rsum2});
    }
    run_pass(p1, {.enable_rewrite_broadcasts = true});

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto bx =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), x);
        auto rsum = add_reduce(
            p2,
            "main:reduce_sum1:main:reduce_sum0_broadcast:main:pointwise0",
            {bx, y},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), rsum1);
                auto mul = add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[1]}, single_pointwise("mul"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_broadcast_pointwise)
{
    // The broadcast only expands an axis that was already size 1 before the
    // reduce, so it hoists completely and the pointwise fuses with the reduce.
    migraphx::shape s1{migraphx::shape::float_type, {2, 1, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 3, 1}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s1);
        auto y     = mm->add_parameter("y", s2);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), rsum);
        auto mul = add_pointwise(p1, "main:pointwise0", {rsumb, y}, single_pointwise("mul"));
        mm->add_return({mul});
    }
    run_pass(p1, {.enable_rewrite_broadcasts = true});

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto bx =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 4}}}), x);
        auto rsum = add_reduce(
            p2,
            "main:reduce_sum0_broadcast:main:pointwise0",
            {bx, y},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                return add_pointwise(
                    p2, rm, "main:pointwise0", {rsum1, inputs[1]}, single_pointwise("mul"));
            });
        mm->add_return({rsum});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(scalar_multibroadcast)
{
    // Matches the find_pointwise_reduce matcher, but input x has a (scalar) shape
    // incompatible with the multibroadcast instruction; therefore it
    // creates a fused_reduce module but does not add a submodule for the
    // multibroadcast instruction.
    migraphx::shape sdot{migraphx::shape::double_type, {80, 204, 204}};
    migraphx::shape sdot_double{migraphx::shape::double_type, {80, 204, 204}};
    migraphx::shape scalar{migraphx::shape::double_type, {1}, {0}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", scalar);
        auto zap = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("sqrt"));
        auto pow = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sdot.lens()}}), zap);
        auto bip = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), pow);

        mm->add_return({bip});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", scalar);
        auto zap = add_pointwise(p2, mm, "main:pointwise0", {x}, single_pointwise("sqrt"));

        auto pow = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sdot.lens()}}), zap);

        // Add a reduce module.  These are created by fuse_reduce::apply() for any reduce
        // instruction whether the individual matchers do anything or not.
        auto* reduce_mod = p2.create_module("main:reduce_sum0");
        auto x0          = reduce_mod->add_parameter("x0", sdot_double);
        auto sqrtbc =
            reduce_mod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), x0);
        reduce_mod->add_return({sqrtbc});

        EXPECT(test::throws([&] {
            mm->add_instruction(
                migraphx::make_op("fused_reduce", {{"axes", {1, 2}}}), {pow}, {reduce_mod});
        }));
        // reduce modules must be flagged for bypass when running subsequent passes
        reduce_mod->set_bypass();
        auto bip = mm->add_instruction(
            migraphx::make_op("fused_reduce", {{"axes", {1, 2}}}), {pow}, {reduce_mod});
        mm->add_return({bip});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(scalar_multibroadcast_contiguous)
{
    // Contains a contiguous op which is not passed through.
    migraphx::shape sdot{migraphx::shape::double_type, {80, 204, 204}};
    migraphx::shape scalar{migraphx::shape::double_type, {1}, {0}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", scalar);
        auto zap = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("sqrt"));
        auto pow = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sdot.lens()}}), zap);
        auto bip    = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), pow);
        auto sqrtbc = mm->add_instruction(migraphx::make_op("contiguous"), bip);

        mm->add_return({sqrtbc});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", scalar);
        auto zap = add_pointwise(p2, mm, "main:pointwise0", {x}, single_pointwise("sqrt"));

        auto pow = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", sdot.lens()}}), zap);

        // Add a reduce module.  These are created by fuse_reduce::apply() for any reduce
        // instruction whether the individual matchers do anything or not.
        auto* reduce_mod = p2.create_module("main:reduce_sum0");

        auto x0 = reduce_mod->add_parameter("x0", sdot);
        auto sqrtbc =
            reduce_mod->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), x0);
        reduce_mod->add_return({sqrtbc});

        EXPECT(test::throws([&] {
            mm->add_instruction(
                migraphx::make_op("fused_reduce", {{"axes", {1, 2}}}), {pow}, {reduce_mod});
        }));
        // reduce modules must be flagged for bypass when running subsequent passes
        reduce_mod->set_bypass();
        auto bip = mm->add_instruction(
            migraphx::make_op("fused_reduce", {{"axes", {1, 2}}}), {pow}, {reduce_mod});
        mm->add_return({bip});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_broadcast_reduce_reshape)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::shape rs{migraphx::shape::float_type, {2, 1}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", rs);
        auto sqrt  = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("sqrt"));
        auto sqrtb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sqrtb);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto add = add_pointwise(p1, "main:pointwise1", {sqrtb, rsumb}, single_pointwise("add"));
        auto reshape = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {6}}}), add);
        mm->add_return({reshape});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", rs);
        auto add = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum0:main:pointwise1",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto sqrt =
                    add_pointwise(p2, rm, "main:pointwise0", inputs, single_pointwise("sqrt"));
                auto sqrtb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
                auto rsum =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), sqrtb);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
                return add_pointwise(
                    p2, rm, "main:pointwise1", {sqrtb, rsumb}, single_pointwise("add"));
            });
        auto reshape = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {6}}}), add);
        mm->add_return({reshape});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_pointwise)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto add = add_pointwise(p1, "main:pointwise0", {rsumb, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto add = add_reduce(
            p2,
            "main:reduce_sum0:main:pointwise0",
            {x, y},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
                return add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[1]}, single_pointwise("add"));
            });
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_pointwise_unfusable_broadcast)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 1, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto y     = mm->add_parameter("y", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), rsum);
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), y);
        auto add = add_pointwise(p1, "main:pointwise0", {rsumb, yb}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto rsum = add_reduce(
            p2, "main:reduce_sum0", {x}, {2}, [&](auto* rm, const auto& inputs, const auto& axes) {
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                           inputs[0]);
            });
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), rsum);
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), y);
        auto add = add_pointwise(p2, "main:pointwise0", {rsumb, yb}, single_pointwise("add"));
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reduce)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto rsumdiff = add_pointwise(p1, "main:pointwise0", {rsumb, x}, single_pointwise("sub"));
        auto rsum2 =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), rsumdiff);
        auto sqrt = add_pointwise(p1, "main:pointwise1", {rsum2}, single_pointwise("sqrt"));
        mm->add_return({sqrt});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto sqrt = add_reduce(
            p2,
            "main:reduce_sum1:main:reduce_sum0:main:pointwise0:main:pointwise1",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
                auto rsumdiff = add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[0]}, single_pointwise("sub"));
                auto rsum2 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 rsumdiff);
                return add_pointwise(p2, rm, "main:pointwise1", {rsum2}, single_pointwise("sqrt"));
            });
        mm->add_return({sqrt});
    }
    EXPECT(p1 == p2);
}

// A reduce over a subset of the axes of another whose inputs are unit along
// the other axes joins it: every output slice holds both reductions
TEST_CASE(reduce_reduce_subset_axes)
{
    migraphx::shape xs{migraphx::shape::float_type, {4, 8, 16}};
    migraphx::shape bs{migraphx::shape::float_type, {4, 8, 1}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", xs);
        auto b    = mm->add_parameter("b", bs);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 2}}}), x);
        auto bsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), b);
        auto add  = add_pointwise(p1, "main:pointwise0", {rsum, bsum}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", xs);
        auto b   = mm->add_parameter("b", bs);
        // The bias sum is copied in first as the input of the fused epilogue
        auto add =
            add_reduce(p2,
                       "main:reduce_sum0:main:pointwise0:main:reduce_sum1",
                       {b, x},
                       {0, 2},
                       [&](auto* rm, const auto& inputs, const auto& axes) {
                           auto bsum = rm->add_instruction(
                               migraphx::make_op("reduce_sum", {{"axes", {0}}}), inputs[0]);
                           auto rsum = rm->add_instruction(
                               migraphx::make_op("reduce_sum", {{"axes", axes}}), inputs[1]);
                           return add_pointwise(
                               p2, rm, "main:pointwise0", {rsum, bsum}, single_pointwise("add"));
                       });
        mm->add_return({add});
    }
    EXPECT(p1.sort() == p2.sort());
}

// A reduce over other axes whose inputs span them stays separate
TEST_CASE(reduce_reduce_different_axes)
{
    migraphx::shape xs{migraphx::shape::float_type, {4, 8, 16}};
    migraphx::shape bs{migraphx::shape::float_type, {4, 8, 16}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", xs);
        auto b    = mm->add_parameter("b", bs);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 2}}}), x);
        auto bsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0}}}), b);
        auto bsb  = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 8, 16}}}), rsum);
        auto add = add_pointwise(p1, "main:pointwise0", {bsb, bsum}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", xs);
        auto b    = mm->add_parameter("b", bs);
        auto rsum = add_reduce(p2, "main:reduce_sum0", {x}, {0, 2}, single_reduce("reduce_sum"));
        auto bsb  = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 8, 16}}}), rsum);
        auto add = add_reduce(
            p2,
            "main:reduce_sum1:main:pointwise0",
            {b, bsb},
            {0},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto bsum = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                return add_pointwise(
                    p2, rm, "main:pointwise0", {inputs[1], bsum}, single_pointwise("add"));
            });
        mm->add_return({add});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_reduce_unfusable_broadcast)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 1, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), rsum);
        auto xb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), x);
        auto rsumdiff = add_pointwise(p1, "main:pointwise0", {rsumb, xb}, single_pointwise("sub"));
        auto rsum2 =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), rsumdiff);
        auto sqrt = add_pointwise(p1, "main:pointwise1", {rsum2}, single_pointwise("sqrt"));
        mm->add_return({sqrt});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p2, "main:reduce_sum0", {x}, {2}, [&](auto* rm, const auto& inputs, const auto& axes) {
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                           inputs[0]);
            });

        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), rsum);
        auto xb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), x);

        auto sqrt = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum1:main:pointwise1",
            {rsumb, xb},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsumdiff = add_pointwise(
                    p2, rm, "main:pointwise0", {inputs[0], inputs[1]}, single_pointwise("sub"));
                auto rsum2 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 rsumdiff);
                return add_pointwise(p2, rm, "main:pointwise1", {rsum2}, single_pointwise("sqrt"));
            });
        mm->add_return({sqrt});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(parallel_reduce_reduce)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", s);
        auto xx     = add_pointwise(p1, "main:pointwise0", {x, x}, single_pointwise("mul"));
        auto rsumx  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsumxx = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), xx);
        auto add = add_pointwise(p1, "main:pointwise1", {rsumx, rsumxx}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto add = add_reduce(
            p2,
            "main:reduce_sum0:main:pointwise1:main:pointwise0:main:reduce_sum1",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto xx = add_pointwise(
                    p2, rm, "main:pointwise0", {inputs[0], inputs[0]}, single_pointwise("mul"));
                auto rsumx = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsumxx =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), xx);
                return add_pointwise(
                    p2, rm, "main:pointwise1", {rsumx, rsumxx}, single_pointwise("add"));
            });
        mm->add_return({add});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(parallel_reduce_reduce_broadcast)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", s);
        auto sqrt   = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("sqrt"));
        auto rsum1  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sqrt);
        auto rsum1b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
        auto relu  = add_pointwise(p1, "main:pointwise1", {x}, single_pointwise("relu"));
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), relu);
        auto add   = add_pointwise(p1, "main:pointwise2", {rsum1, rsum2}, single_pointwise("add"));
        auto addb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), add);
        auto clip =
            add_pointwise(p1, "main:pointwise3", {x, rsum1b, addb}, single_pointwise("clip"));
        mm->add_return({clip});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto clip = add_reduce(
            p2,
            "main:pointwise1:main:reduce_sum1:main:pointwise2:main:pointwise3:main:pointwise0:main:"
            "reduce_sum0",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto&) {
                auto sqrt =
                    add_pointwise(p2, rm, "main:pointwise0", {inputs[0]}, single_pointwise("sqrt"));
                auto rsum1 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sqrt);
                auto rsum1b = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
                auto relu =
                    add_pointwise(p2, rm, "main:pointwise1", {inputs[0]}, single_pointwise("relu"));
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), relu);
                auto add = add_pointwise(
                    p2, rm, "main:pointwise2", {rsum1, rsum2}, single_pointwise("add"));
                auto addb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), add);
                return add_pointwise(
                    p2, rm, "main:pointwise3", {inputs[0], rsum1b, addb}, single_pointwise("clip"));
            });
        mm->add_return({clip});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(parallel_reduce_reduce_broadcast_contiguous)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", s);
        auto sqrt   = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("sqrt"));
        auto rsum1  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sqrt);
        auto rsum1b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
        auto rsum1bc = mm->add_instruction(migraphx::make_op("contiguous"), rsum1b);
        auto relu    = add_pointwise(p1, "main:pointwise1", {x}, single_pointwise("relu"));
        auto rsum2   = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), relu);
        auto add = add_pointwise(p1, "main:pointwise2", {rsum1, rsum2}, single_pointwise("add"));
        auto addb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), add);
        auto clip =
            add_pointwise(p1, "main:pointwise3", {x, rsum1bc, addb}, single_pointwise("clip"));
        mm->add_return({clip});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto clip = add_reduce(
            p2,
            "main:pointwise1:main:reduce_sum1:main:pointwise2:main:pointwise3:main:pointwise0:main:"
            "reduce_sum0",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto&) {
                auto sqrt =
                    add_pointwise(p2, rm, "main:pointwise0", {inputs[0]}, single_pointwise("sqrt"));
                auto rsum1 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), sqrt);
                auto rsum1b = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
                auto relu =
                    add_pointwise(p2, rm, "main:pointwise1", {inputs[0]}, single_pointwise("relu"));
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), relu);
                auto add = add_pointwise(
                    p2, rm, "main:pointwise2", {rsum1, rsum2}, single_pointwise("add"));
                auto addb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), add);
                return add_pointwise(
                    p2, rm, "main:pointwise3", {inputs[0], rsum1b, addb}, single_pointwise("clip"));
            });
        mm->add_return({clip});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_reduce_mismatch_axis)
{
    migraphx::shape s{migraphx::shape::float_type, {4, 2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), rsum1);
        mm->add_return({rsum2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm   = p2.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum1 = add_reduce(p2, "main:reduce_sum0", {x}, {1}, single_reduce("reduce_sum"));
        auto rsum2 = add_reduce(p2, "main:reduce_sum1", {rsum1}, {2}, single_reduce("reduce_sum"));
        mm->add_return({rsum2});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_reduce_broadcast)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto sqrt  = add_pointwise(p1, "main:pointwise0", {rsum1}, single_pointwise("sqrt"));
        auto sqrtb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
        auto add1  = add_pointwise(p1, "main:pointwise1", {sqrtb, x}, single_pointwise("add"));
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), add1);
        auto add2  = add_pointwise(p1, "main:pointwise2", {rsum2, rsum1}, single_pointwise("add"));
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto add2 = add_reduce(
            p2,
            "main:pointwise0:main:pointwise1:main:reduce_sum1:main:pointwise2:main:reduce_sum0",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto sqrt =
                    add_pointwise(p2, rm, "main:pointwise0", {rsum1}, single_pointwise("sqrt"));
                auto sqrtb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
                auto add1 = add_pointwise(
                    p2, rm, "main:pointwise1", {sqrtb, inputs[0]}, single_pointwise("add"));
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add1);
                return add_pointwise(
                    p2, rm, "main:pointwise2", {rsum2, rsum1}, single_pointwise("add"));
            });
        mm->add_return({add2});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(pointwise_reduce_broadcast_contiguous)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto sqrt  = add_pointwise(p1, "main:pointwise0", {rsum1}, single_pointwise("sqrt"));
        auto sqrtb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
        auto sqrtbc = mm->add_instruction(migraphx::make_op("contiguous"), sqrtb);
        auto add1   = add_pointwise(p1, "main:pointwise1", {sqrtbc, x}, single_pointwise("add"));
        auto rsum2  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), add1);
        auto add2   = add_pointwise(p1, "main:pointwise2", {rsum2, rsum1}, single_pointwise("add"));
        mm->add_return({add2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto add2 = add_reduce(
            p2,
            "main:pointwise0:main:pointwise1:main:reduce_sum1:main:pointwise2:main:reduce_sum0",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto sqrt =
                    add_pointwise(p2, rm, "main:pointwise0", {rsum1}, single_pointwise("sqrt"));
                auto sqrtb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
                auto add1 = add_pointwise(
                    p2, rm, "main:pointwise1", {sqrtb, inputs[0]}, single_pointwise("add"));
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add1);
                return add_pointwise(
                    p2, rm, "main:pointwise2", {rsum2, rsum1}, single_pointwise("add"));
            });
        mm->add_return({add2});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reduce_broadcast)
{
    migraphx::shape s{migraphx::shape::float_type, {4, 2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum1 = add_reduce(p1, "test:reduce_sum0", {x}, {1}, single_reduce("reduce_sum"));
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
        auto add = add_reduce(
            p1,
            "test:reduce_sum1",
            {rsumb, x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto add2 =
                    add_pointwise(p1, rm, "test:pointwise0", inputs, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add2);
            });
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p2,
            "test:reduce_sum1:test:reduce_sum0",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
                auto add = add_pointwise(
                    p2, rm, "test:pointwise0", {rsumb, inputs[0]}, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reduce_broadcast_contiguous)
{
    migraphx::shape s{migraphx::shape::float_type, {4, 2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum1 = add_reduce(p1, "test:reduce_sum0", {x}, {1}, single_reduce("reduce_sum"));
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
        auto rsumbc = mm->add_instruction(migraphx::make_op("contiguous"), rsumb);
        auto add    = add_reduce(
            p1,
            "test:reduce_sum1",
            {rsumbc, x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto add2 =
                    add_pointwise(p1, rm, "test:pointwise0", inputs, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add2);
            });
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p2,
            "test:reduce_sum1:test:reduce_sum0",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
                auto add = add_pointwise(
                    p2, rm, "test:pointwise0", {rsumb, inputs[0]}, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), add);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reshape_pointwise1)
{
    migraphx::shape s1{migraphx::shape::float_type, {64, 4}};
    migraphx::shape s2{migraphx::shape::float_type, {8, 8, 2, 2}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s1);
        auto y     = mm->add_parameter("y", s2);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
        auto rsumr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), rsumb);
        auto add = add_pointwise(p1, "main:pointwise0", {rsumr, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto xr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), x);
        auto add = add_reduce(
            p2,
            "main:reduce_sum0_reshape:main:pointwise0",
            {xr, y},
            {2, 3},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), rsum);
                return add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[1]}, single_pointwise("add"));
            });
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reshape_pointwise2)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 32, 40960}};
    migraphx::shape s2{migraphx::shape::float_type, {2, 320, 64, 64}};
    migraphx::shape s3{migraphx::shape::float_type, {2, 32, 10, 64, 64}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s1);
        auto y     = mm->add_parameter("y", s2);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
        auto rsumr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), rsumb);
        auto add = add_pointwise(p1, "main:pointwise0", {rsumr, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto xr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), x);
        auto yr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), y);
        auto add = add_reduce(
            p2,
            "main:reduce_sum0_reshape:main:pointwise0",
            {xr, yr},
            {2, 3, 4},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), rsum);
                return add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[1]}, single_pointwise("add"));
            });
        auto addr = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), add);
        mm->add_return({addr});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_contiguous_reshape_pointwise)
{
    migraphx::shape s1 =
        migraphx::shape::from_permutation(migraphx::shape::float_type, {2, 32, 40960}, {1, 0, 2});
    auto s2 = migraphx::shape{migraphx::shape::float_type, {2, 320, 64, 64}};
    auto s3 = migraphx::shape{migraphx::shape::float_type, {2, 32, 10, 64, 64}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s1);
        auto y     = mm->add_parameter("y", s2);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumc = mm->add_instruction(migraphx::make_op("contiguous"), rsum);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsumc);
        auto rsumr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), rsumb);
        auto add = add_pointwise(p1, "main:pointwise0", {rsumr, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto xr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), x);
        auto yr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), y);
        auto add = add_reduce(
            p2,
            "main:reduce_sum0_reshape:main:pointwise0",
            {xr, yr},
            {2, 3, 4},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), rsum);
                return add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[1]}, single_pointwise("add"));
            });
        auto addr = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), add);
        mm->add_return({addr});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_squeeze_unsqueeze_pointwise1)
{
    migraphx::shape s1{migraphx::shape::float_type, {1, 1, 1, 1, 1, 1, 32, 10, 16, 1, 90, 160}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s1);
        auto rsum =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {7, 8, 9, 10, 11}}}), x);
        auto squeeze = mm->add_instruction(
            migraphx::make_op("squeeze", {{"axes", {1, 2, 3, 4, 5, 7, 8, 9, 10}}}), rsum);
        auto unsqueeze = mm->add_instruction(
            migraphx::make_op("unsqueeze", {{"axes", {1, 2, 3, 4, 5, 7, 8, 10, 11}}}), squeeze);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), unsqueeze);
        auto add = add_pointwise(p1, "main:pointwise0", {rsumb, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s1);
        auto add = add_reduce(
            p2,
            "main:reduce_sum0_reshape:main:pointwise0",
            {x, y},
            {7, 8, 9, 10, 11},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum);
                return add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[1]}, single_pointwise("add"));
            });
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(reduce_reshape_reduce)
{
    migraphx::shape s1{migraphx::shape::float_type, {2, 32, 4096}};
    migraphx::shape s1r{migraphx::shape::float_type, {2, 32, 1}};
    migraphx::shape s2{migraphx::shape::float_type, {4, 16, 64, 64}};
    migraphx::shape s2r{migraphx::shape::float_type, {4, 16, 1, 1}};
    migraphx::shape s3{migraphx::shape::float_type, {2, 2, 16, 64, 64}};
    migraphx::shape s3r{migraphx::shape::float_type, {2, 2, 16, 1, 1}};
    migraphx::program p1;
    {
        auto* mm       = p1.get_main_module();
        auto x1        = mm->add_parameter("x1", s1);
        auto x2        = mm->add_parameter("x2", s1r);
        auto y         = mm->add_parameter("y", s2);
        auto rsum1     = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x1);
        auto rsum1_add = add_pointwise(p1, "main:pointwise0", {rsum1, x2}, single_pointwise("add"));

        auto rsum1_addb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum1_add);
        auto rsum1_sub =
            add_pointwise(p1, "main:pointwise1", {rsum1_addb, x1}, single_pointwise("sub"));
        auto rsum2 =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), rsum1_sub);
        auto rsum2b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum2);
        auto rsum2_sub =
            add_pointwise(p1, "main:pointwise2", {rsum2b, x1}, single_pointwise("sub"));
        auto rsum2_subr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), rsum2_sub);
        auto rsum3 =
            mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2, 3}}}), rsum2_subr);
        auto rsum3b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), rsum3);
        auto rsum3_add = add_pointwise(p1, "main:pointwise3", {rsum3b, y}, single_pointwise("add"));
        mm->add_return({rsum3_add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x1  = mm->add_parameter("x1", s1);
        auto x2  = mm->add_parameter("x2", s1r);
        auto y   = mm->add_parameter("y", s2);
        auto x1r = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), x1);
        auto x2r = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3r.lens()}}), x2);
        auto yr      = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), y);
        auto freduce = add_reduce(
            p2,
            "main:pointwise2:main:reduce_sum2_reshape_reshape:main:pointwise3_reshape:main:reduce_"
            "sum1:main:reduce_sum0:main:pointwise0:main:pointwise1_reshape",
            {x1r, x2r, yr},
            {3, 4},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto add   = add_pointwise(
                    p2, rm, "main:pointwise0", {rsum1, inputs[1]}, single_pointwise("add"));
                auto addb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), add);
                auto sub1 = add_pointwise(
                    p2, rm, "main:pointwise1", {addb, inputs[0]}, single_pointwise("sub"));
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), sub1);
                auto rsum2b = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), rsum2);
                auto sub2 = add_pointwise(
                    p2, rm, "main:pointwise2", {rsum2b, inputs[0]}, single_pointwise("sub"));
                auto rsum3 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), sub2);
                auto rsum3b = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), rsum3);
                return add_pointwise(
                    p2, rm, "main:pointwise3", {rsum3b, inputs[2]}, single_pointwise("add"));
            });
        auto freducer =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), freduce);
        mm->add_return({freducer});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reshape_reduce_reduce_reduce_diff_axes)
{
    migraphx::shape s1{migraphx::shape::float_type, {128, 196, 256}};
    migraphx::shape s2{migraphx::shape::float_type, {1}};
    migraphx::shape s3{migraphx::shape::float_type, {25088, 256}};

    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x1  = mm->add_parameter("x1", s1);
        auto x2  = mm->add_parameter("x2", s1);
        auto l1  = mm->add_literal(migraphx::literal(s2, {1.0}));
        auto l2  = mm->add_literal(migraphx::literal(s2, {2.0}));

        auto x1_rsp = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), x1);
        auto l2_mb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), l2);
        auto l2_ct = mm->add_instruction(migraphx::make_op("contiguous"), l2_mb);
        auto pw0   = add_pointwise(p1, "main:pointwise0", {l2_ct, x1_rsp}, single_pointwise("add"));
        auto pw0_rsp =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s1.lens()}}), pw0);
        auto pw1   = add_pointwise(p1, "main:pointwise1", {x2, pw0_rsp}, single_pointwise("mul"));
        auto pw2   = add_pointwise(p1, "main:pointwise2", {pw1}, single_pointwise("log"));
        auto rsum0 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), pw2);
        auto rsum0_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum0);
        auto rsum0_ct = mm->add_instruction(migraphx::make_op("contiguous"), rsum0_mb);
        auto pw3 =
            add_pointwise(p1, "main:pointwise3", {pw0_rsp, rsum0_ct}, single_pointwise("add"));
        auto pw4    = add_pointwise(p1, "main:pointwise4", {pw3}, single_pointwise("log"));
        auto rsum1  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), pw4);
        auto pw5    = add_pointwise(p1, "main:pointwise5", {rsum1}, single_pointwise("log"));
        auto pw5_mb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), pw5);
        auto pw5_ct = mm->add_instruction(migraphx::make_op("contiguous"), pw5_mb);

        auto l1_mb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), l1);
        auto pw6 = add_pointwise(p1, "main:pointwise6", {pw5_ct, l1_mb}, single_pointwise("mul"));

        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), pw6);
        mm->add_return({rsum2});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x1  = mm->add_parameter("x1", s1);
        auto x2  = mm->add_parameter("x2", s1);
        auto l1  = mm->add_literal(migraphx::literal(s2, {1.0}));
        auto l2  = mm->add_literal(migraphx::literal(s2, {2.0}));

        auto l1_mb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), l1);
        auto l2_mb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), l2);

        auto reduce0 = add_reduce(
            p2,
            "main:pointwise0:main:pointwise1:main:reduce_sum1:main:pointwise2:main:reduce_sum0:"
            "main:pointwise3:main:pointwise4:main:pointwise5:main:pointwise6_reshape",
            {l2_mb, x1, x2, l1_mb},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto pw0 = add_pointwise(
                    p2, rm, "main:pointwise0", {inputs[0], inputs[1]}, single_pointwise("add"));
                auto pw1 = add_pointwise(
                    p2, rm, "main:pointwise1", {inputs[2], pw0}, single_pointwise("mul"));
                auto pw2 = add_pointwise(p2, rm, "main:pointwise2", {pw1}, single_pointwise("log"));
                auto rsum0 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), pw2);
                auto rsum0_mb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), rsum0);
                auto pw3 = add_pointwise(
                    p2, rm, "main:pointwise3", {pw0, rsum0_mb}, single_pointwise("add"));
                auto pw4 = add_pointwise(p2, rm, "main:pointwise4", {pw3}, single_pointwise("log"));
                auto rsum1 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), pw4);
                auto pw5 =
                    add_pointwise(p2, rm, "main:pointwise5", {rsum1}, single_pointwise("log"));
                auto pw5_mb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), pw5);
                return add_pointwise(
                    p2, rm, "main:pointwise6", {pw5_mb, inputs[3]}, single_pointwise("mul"));
            });

        auto reduce1 =
            add_reduce(p2,
                       "main:reduce_sum2",
                       {reduce0},
                       {1},
                       [&](auto* rm, const auto& inputs, const auto& axes) {
                           return rm->add_instruction(
                               migraphx::make_op("reduce_sum", {{"axes", axes}}), inputs[0]);
                       });

        mm->add_return({reduce1});
    }

    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(argmin_fuse)
{
    // Test that argmin with axis=1 gets wrapped in fused_reduce with axes={1}
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto argmin1 = mm->add_instruction(migraphx::make_op("argmin", {{"axis", 1}}), x);
        mm->add_return({argmin1});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm     = p2.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto argmin1 = add_reduce(
            p2, "main:argmin0", {x}, {1}, [](auto* rm, const auto& inputs, const auto& axes) {
                return rm->add_instruction(migraphx::make_op("argmin", {{"axis", axes.front()}}),
                                           inputs);
            });
        mm->add_return({argmin1});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(argmax_fuse)
{
    // Test that argmax with axis=1 gets wrapped in fused_reduce with axes={1}
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto argmax1 = mm->add_instruction(migraphx::make_op("argmax", {{"axis", 1}}), x);
        mm->add_return({argmax1});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm     = p2.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto argmax1 = add_reduce(
            p2, "main:argmax0", {x}, {1}, [](auto* rm, const auto& inputs, const auto& axes) {
                return rm->add_instruction(migraphx::make_op("argmax", {{"axis", axes.front()}}),
                                           inputs);
            });
        mm->add_return({argmax1});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(argmin_axis0)
{
    // Test that argmin with axis=0 gets wrapped in fused_reduce with axes={0}
    migraphx::shape s{migraphx::shape::float_type, {4, 5, 6}};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto argmin1 = mm->add_instruction(migraphx::make_op("argmin", {{"axis", 0}}), x);
        mm->add_return({argmin1});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm     = p2.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto argmin1 = add_reduce(
            p2, "main:argmin0", {x}, {0}, [](auto* rm, const auto& inputs, const auto& axes) {
                return rm->add_instruction(migraphx::make_op("argmin", {{"axis", axes.front()}}),
                                           inputs);
            });
        mm->add_return({argmin1});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(pointwise_argmin)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto y       = mm->add_parameter("y", s);
        auto add     = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto argmin1 = mm->add_instruction(migraphx::make_op("argmin", {{"axis", 1}}), add);
        mm->add_return({argmin1});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm     = p2.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto y       = mm->add_parameter("y", s);
        auto argmin1 = add_reduce(
            p2,
            "main:pointwise0:main:argmin0",
            {x, y},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto add =
                    add_pointwise(p2, rm, "main:pointwise0", inputs, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("argmin", {{"axis", axes.front()}}),
                                           add);
            });
        mm->add_return({argmin1});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(pointwise_argmax)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto y       = mm->add_parameter("y", s);
        auto add     = add_pointwise(p1, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto argmax1 = mm->add_instruction(migraphx::make_op("argmax", {{"axis", 1}}), add);
        mm->add_return({argmax1});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm     = p2.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto y       = mm->add_parameter("y", s);
        auto argmax1 = add_reduce(
            p2,
            "main:pointwise0:main:argmax0",
            {x, y},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto add =
                    add_pointwise(p2, rm, "main:pointwise0", inputs, single_pointwise("add"));
                return rm->add_instruction(migraphx::make_op("argmax", {{"axis", axes.front()}}),
                                           add);
            });
        mm->add_return({argmax1});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(argmin_pointwise)
{
    // argmin → broadcast → pointwise (fusable)
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::shape si{migraphx::shape::int64_type, {2, 1}};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto y       = mm->add_parameter("y", si);
        auto argmin1 = mm->add_instruction(migraphx::make_op("argmin", {{"axis", 1}}), x);
        auto argminb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), argmin1);
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), y);
        auto add = add_pointwise(p1, "main:pointwise0", {argminb, yb}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", si);
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), y);
        auto add = add_reduce(
            p2,
            "main:argmin0:main:pointwise0",
            {x, yb},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto argmin1 = rm->add_instruction(
                    migraphx::make_op("argmin", {{"axis", axes.front()}}), inputs[0]);
                auto argminb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), argmin1);
                return add_pointwise(
                    p2, rm, "main:pointwise0", {argminb, inputs[1]}, single_pointwise("add"));
            });
        mm->add_return({add});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(argmin_pointwise_unfusable_broadcast)
{
    // argmin → broadcast to different dims → pointwise (unfusable)
    migraphx::shape s{migraphx::shape::float_type, {2, 1, 3}};
    migraphx::shape si{migraphx::shape::int64_type, {2, 1, 1}};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto y       = mm->add_parameter("y", si);
        auto argmin1 = mm->add_instruction(migraphx::make_op("argmin", {{"axis", 2}}), x);
        auto argminb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), argmin1);
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), y);
        auto add = add_pointwise(p1, "main:pointwise0", {argminb, yb}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm     = p2.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto y       = mm->add_parameter("y", si);
        auto argmin1 = add_reduce(
            p2, "main:argmin0", {x}, {2}, [](auto* rm, const auto& inputs, const auto& axes) {
                return rm->add_instruction(migraphx::make_op("argmin", {{"axis", axes.front()}}),
                                           inputs);
            });
        auto argminb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), argmin1);
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), y);
        auto add = add_pointwise(p2, "main:pointwise0", {argminb, yb}, single_pointwise("add"));
        mm->add_return({add});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_argmin)
{
    // reduce_sum → broadcast → pointwise → argmin (fusable two-reduce pattern)
    migraphx::shape s{migraphx::shape::float_type, {2, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto sub     = add_pointwise(p1, "main:pointwise0", {rsumb, x}, single_pointwise("sub"));
        auto argmin1 = mm->add_instruction(migraphx::make_op("argmin", {{"axis", 1}}), sub);
        mm->add_return({argmin1});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm     = p2.get_main_module();
        auto x       = mm->add_parameter("x", s);
        auto argmin1 = add_reduce(
            p2,
            "main:argmin1:main:reduce_sum0:main:pointwise0",
            {x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
                auto sub = add_pointwise(
                    p2, rm, "main:pointwise0", {rsumb, inputs[0]}, single_pointwise("sub"));
                return rm->add_instruction(migraphx::make_op("argmin", {{"axis", axes.front()}}),
                                           sub);
            });
        mm->add_return({argmin1});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_argmin_unfusable_broadcast)
{
    // reduce_sum → broadcast to different dims → pointwise → argmin (unfusable)
    migraphx::shape s{migraphx::shape::float_type, {2, 1, 3}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), rsum);
        auto xb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), x);
        auto sub     = add_pointwise(p1, "main:pointwise0", {rsumb, xb}, single_pointwise("sub"));
        auto argmin1 = mm->add_instruction(migraphx::make_op("argmin", {{"axis", 2}}), sub);
        mm->add_return({argmin1});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p2, "main:reduce_sum0", {x}, {2}, [](auto* rm, const auto& inputs, const auto& axes) {
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                           inputs);
            });
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), rsum);
        auto xb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 4, 3}}}), x);
        auto argmin1 = add_reduce(
            p2,
            "main:pointwise0:main:argmin1",
            {rsumb, xb},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto sub = add_pointwise(
                    p2, rm, "main:pointwise0", {inputs[0], inputs[1]}, single_pointwise("sub"));
                return rm->add_instruction(migraphx::make_op("argmin", {{"axis", axes.front()}}),
                                           sub);
            });
        mm->add_return({argmin1});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(argmin_reshape_pointwise)
{
    // Test that argmin axis gets transformed correctly when reshaped
    // Input: {64, 4} with argmin on axis 1
    // Reshape to: {8, 8, 2, 2} means axis 1 splits into axes {2, 3}
    // After reshape, argmin should use axis 2 (first of the new axes)
    migraphx::shape s1{migraphx::shape::float_type, {64, 4}};
    migraphx::shape s2{migraphx::shape::int64_type, {8, 8, 2, 2}};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto x       = mm->add_parameter("x", s1);
        auto y       = mm->add_parameter("y", s2);
        auto argmin  = mm->add_instruction(migraphx::make_op("argmin", {{"axis", 1}}), x);
        auto argminb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), argmin);
        auto argminr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), argminb);
        auto add = add_pointwise(p1, "main:pointwise0", {argminr, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto xr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), x);
        // argmin in fused_reduce with axes={2,3}, but argmin only uses axis=2
        auto argmin_reduce =
            add_reduce(p2,
                       "main:argmin0_reshape",
                       {xr},
                       {2, 3},
                       [&](auto* rm, const auto& inputs, const auto& axes) {
                           return rm->add_instruction(
                               migraphx::make_op("argmin", {{"axis", axes.front()}}), inputs[0]);
                       });
        // Output shape is {8,8,1,2} - only axis 2 reduced, axis 3 remains
        auto argminb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s2.lens()}}), argmin_reduce);
        auto add = add_pointwise(p2, "main:pointwise0", {argminb, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(argmax_reshape_pointwise)
{
    // Test that argmax axis gets transformed correctly when reshaped
    migraphx::shape s1{migraphx::shape::float_type, {2, 32, 40960}};
    migraphx::shape s2{migraphx::shape::int64_type, {2, 320, 64, 64}};
    migraphx::shape s3{migraphx::shape::float_type, {2, 32, 10, 64, 64}};
    migraphx::shape s3i{migraphx::shape::int64_type, {2, 32, 10, 64, 64}};
    migraphx::program p1;
    {
        auto* mm     = p1.get_main_module();
        auto x       = mm->add_parameter("x", s1);
        auto y       = mm->add_parameter("y", s2);
        auto argmax  = mm->add_instruction(migraphx::make_op("argmax", {{"axis", 2}}), x);
        auto argmaxb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s1.lens()}}), argmax);
        auto argmaxr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), argmaxb);
        auto add = add_pointwise(p1, "main:pointwise0", {argmaxr, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s1);
        auto y   = mm->add_parameter("y", s2);
        auto xr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3.lens()}}), x);
        // argmax in fused_reduce with axes={2,3,4}, but argmax only uses axis=2
        auto argmax_reduce =
            add_reduce(p2,
                       "main:argmax0_reshape",
                       {xr},
                       {2, 3, 4},
                       [&](auto* rm, const auto& inputs, const auto& axes) {
                           return rm->add_instruction(
                               migraphx::make_op("argmax", {{"axis", axes.front()}}), inputs[0]);
                       });
        // Output shape is {2,32,1,64,64} - only axis 2 reduced
        auto argmaxb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s3.lens()}}), argmax_reduce);
        auto yr   = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s3i.lens()}}), y);
        auto add  = add_pointwise(p2, "main:pointwise0", {argmaxb, yr}, single_pointwise("add"));
        auto addr = mm->add_instruction(migraphx::make_op("reshape", {{"dims", s2.lens()}}), add);
        mm->add_return({addr});
    }
    EXPECT(p1.sort() == p2.sort());
}

static auto convert_mul_pointwise()
{
    return [](auto* pm, const auto& inputs) {
        auto cvt = pm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}),
            inputs[0]);
        return pm->add_instruction(migraphx::make_op("mul"), cvt, inputs[1]);
    };
}

TEST_CASE(unpack_int4_reduce)
{
    migraphx::shape ps{migraphx::shape::uint8_type, {2, 3, 4}};
    migraphx::shape xs{migraphx::shape::float_type, {2, 3, 8}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto packed = mm->add_parameter("wp", ps);
        auto x      = mm->add_parameter("x", xs);
        auto up     = mm->add_instruction(migraphx::make_op("unpack_int4"), packed);
        auto mul    = add_pointwise(p1, "main:pointwise0", {up, x}, convert_mul_pointwise());
        auto rsum   = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), mul);
        mm->add_return({rsum});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm    = p2.get_main_module();
        auto packed = mm->add_parameter("wp", ps);
        auto x      = mm->add_parameter("x", xs);
        auto rsum   = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum0:unpack_int4",
            {packed, x},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto up  = rm->add_instruction(migraphx::make_op("unpack_int4"), inputs[0]);
                auto mul = add_pointwise(
                    p2, rm, "main:pointwise0", {up, inputs[1]}, convert_mul_pointwise());
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(unpack_int4_reshape_reduce)
{
    migraphx::shape ps{migraphx::shape::uint8_type, {2, 12}};
    migraphx::shape xs{migraphx::shape::float_type, {2, 3, 8}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto packed = mm->add_parameter("wp", ps);
        auto x      = mm->add_parameter("x", xs);
        auto up     = mm->add_instruction(migraphx::make_op("unpack_int4"), packed);
        auto up_reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 8}}}), up);
        auto mul  = add_pointwise(p1, "main:pointwise0", {up_reshape, x}, convert_mul_pointwise());
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), mul);
        mm->add_return({rsum});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm    = p2.get_main_module();
        auto packed = mm->add_parameter("wp", ps);
        auto x      = mm->add_parameter("x", xs);
        auto packed_reshape =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 4}}}), packed);
        auto rsum = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum0:unpack_int4",
            {packed_reshape, x},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto up =
                    rm->add_instruction(migraphx::make_op("unpack_int4", {{"axis", 2}}), inputs[0]);
                auto mul = add_pointwise(
                    p2, rm, "main:pointwise0", {up, inputs[1]}, convert_mul_pointwise());
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(unpack_int4_unreduced_axis)
{
    migraphx::shape ps{migraphx::shape::uint8_type, {2, 3, 4}};
    migraphx::shape xs{migraphx::shape::float_type, {2, 3, 8}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto packed = mm->add_parameter("wp", ps);
        auto x      = mm->add_parameter("x", xs);
        auto up     = mm->add_instruction(migraphx::make_op("unpack_int4"), packed);
        auto mul    = add_pointwise(p1, "main:pointwise0", {up, x}, convert_mul_pointwise());
        auto rsum   = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        mm->add_return({rsum});
    }
    run_pass(p1);

    // The unpack axis is not reduced so the unpack stays outside
    migraphx::program p2;
    {
        auto* mm    = p2.get_main_module();
        auto packed = mm->add_parameter("wp", ps);
        auto x      = mm->add_parameter("x", xs);
        auto up     = mm->add_instruction(migraphx::make_op("unpack_int4"), packed);
        auto rsum   = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum0",
            {up, x},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto mul = add_pointwise(
                    p2, rm, "main:pointwise0", {inputs[0], inputs[1]}, convert_mul_pointwise());
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(pointwise_reshapes_reduce_shadowed_broadcast)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 8}};
    migraphx::shape ws{migraphx::shape::float_type, {4, 2, 4}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", xs);
        auto w   = mm->add_parameter("w", ws);
        auto pw0 = add_pointwise(p1, "main:pointwise0", {x}, single_pointwise("sqrt"));
        auto xu  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), pw0);
        auto xb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 4, 8}}}), xu);
        auto pw1  = add_pointwise(p1, "main:pointwise1", {w}, single_pointwise("exp"));
        auto wu   = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), pw1);
        auto wr   = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 4, 8}}}), wu);
        auto mul  = add_pointwise(p1, "main:pointwise2", {xb, wr}, single_pointwise("mul"));
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), mul);
        mm->add_return({rsum});
    }
    run_pass(p1);

    // The x chain broadcasts an unreduced axis so it cant be rewritten; it
    // must not shadow the rewritable reshape chain on the w input
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", xs);
        auto w   = mm->add_parameter("w", ws);
        auto pw0 = add_pointwise(p2, "main:pointwise0", {x}, single_pointwise("sqrt"));
        auto wu  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), w);
        auto xr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 2, 4}}}), pw0);
        auto xb  = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {1, 4, 2, 4}}}), xr);
        auto rsum = add_reduce(
            p2,
            "main:pointwise1:main:pointwise2:main:reduce_sum0_reshape",
            {wu, xb},
            {2, 3},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto exp =
                    add_pointwise(p2, rm, "main:pointwise1", {inputs[0]}, single_pointwise("exp"));
                auto mul = add_pointwise(
                    p2, rm, "main:pointwise2", {inputs[1], exp}, single_pointwise("mul"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
            });
        auto sq = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), rsum);
        mm->add_return({sq});
    }
    EXPECT(p1.sort() == p2.sort());
}

static auto dequant_mul_pointwise()
{
    return [](auto* pm, const auto& inputs) {
        auto w = pm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}),
            inputs[0]);
        auto zp = pm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}),
            inputs[1]);
        auto sub = pm->add_instruction(migraphx::make_op("sub"), w, zp);
        return pm->add_instruction(migraphx::make_op("mul"), sub, inputs[2]);
    };
}

// Select nibble x1 (16 for the low nibble, 1 for the high) of byte x0
static auto nibble_pointwise()
{
    return [](auto* pm, const auto& inputs) {
        migraphx::shape s{migraphx::shape::uint8_type};
        auto sixteen = pm->add_literal(migraphx::literal{s, {16}});
        auto fifteen = pm->add_literal(migraphx::literal{s, {15}});
        auto shifted = pm->add_instruction(migraphx::make_op("mul"), inputs[0], inputs[1]);
        auto high    = pm->add_instruction(migraphx::make_op("div"), shifted, sixteen);
        return pm->add_instruction(migraphx::make_op("bitwise_and"), high, fifteen);
    };
}

TEST_CASE(unpack_int4_broadcast_reduce)
{
    migraphx::shape ws{migraphx::shape::uint8_type, {4, 2, 4}};
    migraphx::shape zs{migraphx::shape::uint8_type, {4, 1}};
    migraphx::shape xs{migraphx::shape::float_type, {1, 2, 8}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto wp  = mm->add_parameter("wp", ws);
        auto zpp = mm->add_parameter("zpp", zs);
        auto x   = mm->add_parameter("x", xs);
        auto w   = mm->add_instruction(migraphx::make_op("unpack_int4"), wp);
        auto zp  = mm->add_instruction(migraphx::make_op("unpack_int4"), zpp);
        auto zpb = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {4, 2, 8}}}), zp);
        auto xb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 2, 8}}}), x);
        auto mul  = add_pointwise(p1, "main:pointwise0", {w, zpb, xb}, dequant_mul_pointwise());
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), mul);
        mm->add_return({rsum});
    }
    run_pass(p1);

    // The zero point is broadcast over the block elements so its unpack axis
    // is not the vectorized axis: the axis is split in two so the packed
    // bytes are a view broadcast over both nibbles and a pointwise selects
    // the nibble
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto wp  = mm->add_parameter("wp", ws);
        auto zpp = mm->add_parameter("zpp", zs);
        auto x   = mm->add_parameter("x", xs);
        auto zpb = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {4, 1, 2, 8}}}), zpp);
        auto sel = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::uint8_type, {2}}, {16, 1}});
        auto selu = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), sel);
        auto selb = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {4, 1, 2, 8}}}), selu);
        auto wpr = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 1, 2, 4}}}), wp);
        auto xu  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto xb  = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {4, 1, 2, 8}}}), xu);
        auto rsum = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum0:unpack_int4:unpack_int4",
            {wpr, zpb, xb, selb},
            {1, 2, 3},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto zp =
                    add_pointwise(p2,
                                  rm,
                                  "main:pointwise0:main:reduce_sum0:unpack_int4:unpack_int4:nibble",
                                  {inputs[1], inputs[3]},
                                  nibble_pointwise());
                auto w =
                    rm->add_instruction(migraphx::make_op("unpack_int4", {{"axis", 3}}), inputs[0]);
                auto mul = add_pointwise(
                    p2, rm, "main:pointwise0", {w, zp, inputs[2]}, dequant_mul_pointwise());
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
            });
        auto sq = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), rsum);
        mm->add_return({sq});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(unpack_int4_broadcast_reduce_pointwise)
{
    migraphx::shape ws{migraphx::shape::uint8_type, {4, 2, 4}};
    migraphx::shape zs{migraphx::shape::uint8_type, {4, 1}};
    migraphx::shape xs{migraphx::shape::float_type, {1, 2, 8}};
    migraphx::shape bs{migraphx::shape::float_type, {4, 1, 1}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto wp  = mm->add_parameter("wp", ws);
        auto zpp = mm->add_parameter("zpp", zs);
        auto x   = mm->add_parameter("x", xs);
        auto b   = mm->add_parameter("b", bs);
        auto w   = mm->add_instruction(migraphx::make_op("unpack_int4"), wp);
        auto zp  = mm->add_instruction(migraphx::make_op("unpack_int4"), zpp);
        auto zpb = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {4, 2, 8}}}), zp);
        auto xb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 2, 8}}}), x);
        auto mul  = add_pointwise(p1, "main:pointwise0", {w, zpb, xb}, dequant_mul_pointwise());
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1, 2}}}), mul);
        auto add  = add_pointwise(p1, "main:pointwise1", {rsum, b}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);

    // The epilogue input at the output shape is unit along the split axis
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto wp  = mm->add_parameter("wp", ws);
        auto zpp = mm->add_parameter("zpp", zs);
        auto x   = mm->add_parameter("x", xs);
        auto b   = mm->add_parameter("b", bs);
        auto zpb = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 0}, {"out_lens", {4, 1, 2, 8}}}), zpp);
        auto sel = mm->add_literal(
            migraphx::literal{migraphx::shape{migraphx::shape::uint8_type, {2}}, {16, 1}});
        auto selu = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0}}}), sel);
        auto selb = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", {4, 1, 2, 8}}}), selu);
        auto wpr = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 1, 2, 4}}}), wp);
        auto xu  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto xb  = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", {4, 1, 2, 8}}}), xu);
        auto br   = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 1, 1, 1}}}), b);
        auto rsum = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum0:main:pointwise1:unpack_int4:unpack_int4",
            {wpr, zpb, xb, br, selb},
            {1, 2, 3},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto zp = add_pointwise(p2,
                                        rm,
                                        "main:pointwise0:main:reduce_sum0:main:pointwise1:unpack_"
                                        "int4:unpack_int4:nibble",
                                        {inputs[1], inputs[4]},
                                        nibble_pointwise());
                auto w =
                    rm->add_instruction(migraphx::make_op("unpack_int4", {{"axis", 3}}), inputs[0]);
                auto mul = add_pointwise(
                    p2, rm, "main:pointwise0", {w, zp, inputs[2]}, dequant_mul_pointwise());
                auto rs =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
                return add_pointwise(
                    p2, rm, "main:pointwise1", {rs, inputs[3]}, single_pointwise("add"));
            });
        auto sq = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), rsum);
        mm->add_return({sq});
    }
    EXPECT(p1.sort() == p2.sort());
}

// The gather of the rows selected by the indices moves into the reduce,
// which reads the selected rows of the data in place
TEST_CASE(gather_reduce)
{
    migraphx::shape ws{migraphx::shape::float_type, {8, 3, 8}};
    migraphx::shape is{migraphx::shape::int32_type, {2}};
    migraphx::shape xs{migraphx::shape::float_type, {2, 3, 8}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto w    = mm->add_parameter("w", ws);
        auto idx  = mm->add_parameter("idx", is);
        auto x    = mm->add_parameter("x", xs);
        auto g    = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto mul  = add_pointwise(p1, "main:pointwise0", {g, x}, single_pointwise("mul"));
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), mul);
        mm->add_return({rsum});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto w    = mm->add_parameter("w", ws);
        auto idx  = mm->add_parameter("idx", is);
        auto x    = mm->add_parameter("x", xs);
        auto rsum = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum0:gather",
            {w, idx, x},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto g = rm->add_instruction(
                    migraphx::make_op("gather", {{"axis", 0}}), inputs[0], inputs[1]);
                auto mul = add_pointwise(
                    p2, rm, "main:pointwise0", {g, inputs[2]}, single_pointwise("mul"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1.sort() == p2.sort());
}

// The gather moves through the views to the reduce: the data is viewed the
// same way with the gather axis at the data length, including a leading
// unit dim, and the gathers share the indices input
TEST_CASE(gather_reshape_reduce)
{
    migraphx::shape ws{migraphx::shape::float_type, {8, 3, 8}};
    migraphx::shape bs{migraphx::shape::float_type, {8, 3}};
    migraphx::shape is{migraphx::shape::int32_type, {2}};
    migraphx::shape xs{migraphx::shape::float_type, {1, 2, 3, 2, 4}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto w   = mm->add_parameter("w", ws);
        auto b   = mm->add_parameter("b", bs);
        auto idx = mm->add_parameter("idx", is);
        auto x   = mm->add_parameter("x", xs);
        auto g   = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto gr = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 2, 3, 2, 4}}}), g);
        auto gb = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), b, idx);
        auto gbu  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 3, 4}}}), gb);
        auto mul  = add_pointwise(p1, "main:pointwise0", {gr, x}, single_pointwise("mul"));
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3, 4}}}), mul);
        auto add  = add_pointwise(p1, "main:pointwise1", {rsum, gbu}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto w   = mm->add_parameter("w", ws);
        auto b   = mm->add_parameter("b", bs);
        auto idx = mm->add_parameter("idx", is);
        auto x   = mm->add_parameter("x", xs);
        auto wr = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 8, 3, 2, 4}}}), w);
        auto br  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {0, 3, 4}}}), b);
        auto add =
            add_reduce(p2,
                       "main:pointwise0:main:reduce_sum0:main:pointwise1:gather",
                       {wr, idx, br, x},
                       {3, 4},
                       [&](auto* rm, const auto& inputs, const auto& axes) {
                           auto g = rm->add_instruction(
                               migraphx::make_op("gather", {{"axis", 1}}), inputs[0], inputs[1]);
                           auto gb = rm->add_instruction(
                               migraphx::make_op("gather", {{"axis", 1}}), inputs[2], inputs[1]);
                           auto mul = add_pointwise(
                               p2, rm, "main:pointwise0", {g, inputs[3]}, single_pointwise("mul"));
                           auto rsum = rm->add_instruction(
                               migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
                           return add_pointwise(
                               p2, rm, "main:pointwise1", {rsum, gb}, single_pointwise("add"));
                       });
        mm->add_return({add});
    }
    EXPECT(p1.sort() == p2.sort());
}

// A gather broadcast over a reduced axis, such as per-row scales, is viewed
// from the data broadcast the same way
TEST_CASE(gather_broadcast_reduce)
{
    migraphx::shape ws{migraphx::shape::float_type, {8, 3, 8}};
    migraphx::shape ss{migraphx::shape::float_type, {8, 3, 1}};
    migraphx::shape is{migraphx::shape::int32_type, {2}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto w   = mm->add_parameter("w", ws);
        auto s   = mm->add_parameter("s", ss);
        auto idx = mm->add_parameter("idx", is);
        auto g   = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), w, idx);
        auto gs  = mm->add_instruction(migraphx::make_op("gather", {{"axis", 0}}), s, idx);
        auto gsb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {2, 3, 8}}}), gs);
        auto mul  = add_pointwise(p1, "main:pointwise0", {g, gsb}, single_pointwise("mul"));
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), mul);
        mm->add_return({rsum});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto w   = mm->add_parameter("w", ws);
        auto s   = mm->add_parameter("s", ss);
        auto idx = mm->add_parameter("idx", is);
        auto sb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8, 3, 8}}}), s);
        auto rsum = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum0:gather",
            {w, idx, sb},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto g = rm->add_instruction(
                    migraphx::make_op("gather", {{"axis", 0}}), inputs[0], inputs[1]);
                auto gs = rm->add_instruction(
                    migraphx::make_op("gather", {{"axis", 0}}), inputs[2], inputs[1]);
                auto mul =
                    add_pointwise(p2, rm, "main:pointwise0", {g, gs}, single_pointwise("mul"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
            });
        mm->add_return({rsum});
    }
    EXPECT(p1.sort() == p2.sort());
}

// A gather along the fastest axis, which the kernel reads in vectors, or
// with a single index, whose axis would merge away, stays outside
TEST_CASE(gather_reduce_unfused)

{
    migraphx::shape ws{migraphx::shape::float_type, {2, 3, 8}};
    auto create = [&](const migraphx::shape& is, int axis, const migraphx::shape& xs) {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto w    = mm->add_parameter("w", ws);
        auto idx  = mm->add_parameter("idx", is);
        auto x    = mm->add_parameter("x", xs);
        auto g    = mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), w, idx);
        auto mul  = add_pointwise(p, "main:pointwise0", {g, x}, single_pointwise("mul"));
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), mul);
        mm->add_return({rsum});
        return p;
    };
    auto expected = [&](const migraphx::shape& is, int axis, const migraphx::shape& xs) {
        migraphx::program p;
        auto* mm  = p.get_main_module();
        auto w    = mm->add_parameter("w", ws);
        auto idx  = mm->add_parameter("idx", is);
        auto x    = mm->add_parameter("x", xs);
        auto g    = mm->add_instruction(migraphx::make_op("gather", {{"axis", axis}}), w, idx);
        auto rsum = add_reduce(
            p,
            "main:pointwise0:main:reduce_sum0",
            {g, x},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto mul = add_pointwise(
                    p, rm, "main:pointwise0", {inputs[0], inputs[1]}, single_pointwise("mul"));
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
            });
        mm->add_return({rsum});
        return p;
    };
    migraphx::shape is8{migraphx::shape::int32_type, {8}};
    migraphx::shape xs8{migraphx::shape::float_type, {2, 3, 8}};
    auto p1 = create(is8, 2, xs8);
    run_pass(p1);
    EXPECT(p1.sort() == expected(is8, 2, xs8).sort());
    migraphx::shape is1{migraphx::shape::int32_type, {1}};
    migraphx::shape xs1{migraphx::shape::float_type, {1, 3, 8}};
    auto p2 = create(is1, 0, xs1);
    run_pass(p2);
    EXPECT(p2.sort() == expected(is1, 0, xs1).sort());
}

// The slices cut the interleaved halves of a reduce output axis that a
// reshape split in two: the axis is split the same way on every input and in
// the submodule, then the slices split the reduce and the pointwise over
// both halves fuses as the epilogue of the merged reduces
TEST_CASE(reduce_slice_interleaved)

{
    migraphx::shape ws{migraphx::shape::float_type, {8, 4}};
    migraphx::shape xs{migraphx::shape::float_type, {1, 4}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto w   = mm->add_parameter("w", ws);
        auto x   = mm->add_parameter("x", xs);
        auto xb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {8, 4}}}), x);
        auto mul   = add_pointwise(p1, "main:pointwise0", {w, xb}, single_pointwise("mul"));
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        auto pairs = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 2}}}), rsum);
        auto gate  = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), pairs);
        auto up = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), pairs);
        auto add = add_pointwise(p1, "main:pointwise1", {gate, up}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto w   = mm->add_parameter("w", ws);
        auto x   = mm->add_parameter("x", xs);
        auto wr  = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 2, 4}}}), w);
        auto w0  = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), wr);
        auto w1 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), wr);
        auto xr = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), x);
        auto xb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 2, 4}}}), xr);
        auto x0 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {0}}, {"ends", {1}}}), xb);
        auto x1 = mm->add_instruction(
            migraphx::make_op("slice", {{"axes", {1}}, {"starts", {1}}, {"ends", {2}}}), xb);
        // The up half is merged into the gate half carrying the epilogue, and
        // both halves share the pointwise module of the products
        auto add = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum0_split_slice0:main:pointwise1:main:pointwise0:main:"
            "reduce_sum0_split_slice1",
            {w1, x1, w0, x0},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto* pm = create_pointwise_module(
                    p2, "main:pointwise0", {inputs[0], inputs[1]}, single_pointwise("mul"));
                auto mul1 = rm->add_instruction(
                    migraphx::make_op("pointwise"), {inputs[0], inputs[1]}, {pm});
                auto rsum1 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul1);
                auto mul0 = rm->add_instruction(
                    migraphx::make_op("pointwise"), {inputs[2], inputs[3]}, {pm});
                auto rsum0 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul0);
                return add_pointwise(
                    p2, rm, "main:pointwise1", {rsum0, rsum1}, single_pointwise("add"));
            });
        auto sq = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {2}}}), add);
        mm->add_return({sq});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_slice_pointwise)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 8}};
    migraphx::shape ws{migraphx::shape::float_type, {4, 8}};
    auto slice_op = [](int64_t start, int64_t end) {
        return migraphx::make_op("slice", {{"axes", {0}}, {"starts", {start}}, {"ends", {end}}});
    };
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", xs);
        auto w   = mm->add_parameter("w", ws);
        auto xb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 8}}}), x);
        auto mul  = add_pointwise(p1, "main:pointwise0", {w, xb}, single_pointwise("mul"));
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), mul);
        auto sq   = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), rsum);
        auto a    = mm->add_instruction(slice_op(0, 2), sq);
        auto b    = mm->add_instruction(slice_op(2, 4), sq);
        auto add  = add_pointwise(p1, "main:pointwise1", {a, b}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", xs);
        auto w   = mm->add_parameter("w", ws);
        auto xb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {4, 8}}}), x);
        auto wb  = mm->add_instruction(slice_op(2, 4), w);
        auto xbb = mm->add_instruction(slice_op(2, 4), xb);
        auto wa  = mm->add_instruction(slice_op(0, 2), w);
        auto xba = mm->add_instruction(slice_op(0, 2), xb);
        auto* pm0 =
            create_pointwise_module(p2, "main:pointwise0", {w, xb}, single_pointwise("mul"));
        auto rsum = add_reduce(
            p2,
            "main:pointwise0:main:reduce_sum0_slice0_2:main:pointwise1:main:pointwise0:main:"
            "reduce_sum0_slice2_4",
            {wb, xbb, wa, xba},
            {1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto mul_b = rm->add_instruction(
                    migraphx::make_op("pointwise"), {inputs[0], inputs[1]}, {pm0});
                auto rb =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul_b);
                auto mul_a = rm->add_instruction(
                    migraphx::make_op("pointwise"), {inputs[2], inputs[3]}, {pm0});
                auto ra =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul_a);
                return add_pointwise(p2, rm, "main:pointwise1", {ra, rb}, single_pointwise("add"));
            });
        auto sq = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), rsum);
        mm->add_return({sq});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_squeeze_pointwise)
{
    migraphx::shape xs{migraphx::shape::float_type, {4, 8}};
    migraphx::shape ys{migraphx::shape::float_type, {4}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", xs);
        auto y    = mm->add_parameter("y", ys);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {1}}}), x);
        auto sq   = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), rsum);
        auto add  = add_pointwise(p1, "main:pointwise0", {sq, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", xs);
        auto y   = mm->add_parameter("y", ys);
        auto yu  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {1}}}), y);
        auto rsum =
            add_reduce(p2,
                       "main:reduce_sum0_reshape:main:pointwise0",
                       {x, yu},
                       {1},
                       [&](auto* rm, const auto& inputs, const auto& axes) {
                           auto rs = rm->add_instruction(
                               migraphx::make_op("reduce_sum", {{"axes", axes}}), inputs[0]);
                           return add_pointwise(
                               p2, rm, "main:pointwise0", {rs, inputs[1]}, single_pointwise("add"));
                       });
        auto sq = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), rsum);
        mm->add_return({sq});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_squeeze_all_pointwise_scalar)
{
    // Squeezing every axis of the reduce clamps the rank to 1, so the {1}
    // input cant be unsqueezed back to the reduce shape and is broadcast instead
    migraphx::shape xs{migraphx::shape::float_type, {3, 4}};
    migraphx::shape ys{migraphx::shape::float_type, {1}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", xs);
        auto y    = mm->add_parameter("y", ys);
        auto rmax = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {0, 1}}}), x);
        auto sq   = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1}}}), rmax);
        auto add  = add_pointwise(p1, "main:pointwise0", {sq, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", xs);
        auto y   = mm->add_parameter("y", ys);
        auto yb =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", {1, 1}}}), y);
        auto* pm0 = create_pointwise_module(p2, "main:pointwise0", {x, y}, single_pointwise("add"));
        auto rmax = add_reduce(
            p2,
            "main:reduce_max0:main:pointwise0",
            {x, yb},
            {0, 1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto r = rm->add_instruction(migraphx::make_op("reduce_max", {{"axes", axes}}),
                                             inputs[0]);
                return rm->add_instruction(migraphx::make_op("pointwise"), {r, inputs[1]}, {pm0});
            });
        auto sq = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1}}}), rmax);
        mm->add_return({sq});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(reduce_reshape_squeeze_all_pointwise)
{
    // Two fully reduced outputs, one behind a reshape and one behind a squeeze,
    // combined by a pointwise: the epilogue fusion must broadcast the {1} reshape
    // output and rewrite_reshapes must not rebase across the 1-element reduce
    migraphx::shape s{migraphx::shape::float_type, {3, 2}};
    migraphx::program p1;
    {
        auto* mm = p1.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto y   = mm->add_parameter("y", s);
        auto r1  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1}}}), x);
        auto rsh = mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1}}}), r1);
        auto r2  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 1}}}), y);
        auto sq  = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1}}}), r2);
        auto add = add_pointwise(p1, "main:pointwise0", {rsh, sq}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto y    = mm->add_parameter("y", s);
        auto rsum = add_reduce(
            p2,
            "main:reduce_sum1:main:pointwise0:main:reduce_sum0",
            {x, y},
            {0, 1},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto r1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                              inputs[0]);
                auto r2 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                              inputs[1]);
                return add_pointwise(p2, rm, "main:pointwise0", {r1, r2}, single_pointwise("add"));
            });
        auto sq = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {0, 1}}}), rsum);
        mm->add_return({sq});
    }
    EXPECT(p1.sort() == p2.sort());
}

// The reduce output is reshaped, transposed and squeezed before the
// pointwise. The reduce is rebuilt at the common dims with its reduced axes
// kept, so the pointwise fuses into it and the squeeze moves after
TEST_CASE(reduce_reshape_transpose_pointwise)
{
    migraphx::shape xs{migraphx::shape::float_type, {1, 1, 8, 2, 4}};
    migraphx::shape ys{migraphx::shape::float_type, {1, 2, 1, 4}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", xs);
        auto y    = mm->add_parameter("y", ys);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {3, 4}}}), x);
        auto rsumr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 2, 4, 1, 1}}}), rsum);
        auto rsumt = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3, 4, 5}}}), rsumr);
        auto rsums = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {4, 5}}}), rsumt);
        auto add   = add_pointwise(p1, "main:pointwise0", {rsums, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", xs);
        auto y   = mm->add_parameter("y", ys);
        auto xr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 2, 4, 2, 4}}}), x);
        auto xt = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3, 4, 5}}}), xr);
        auto yu = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {4, 5}}}), y);
        auto add =
            add_reduce(p2,
                       "main:reduce_sum0_reshape:main:pointwise0",
                       {xt, yu},
                       {4, 5},
                       [&](auto* rm, const auto& inputs, const auto& axes) {
                           auto rs = rm->add_instruction(
                               migraphx::make_op("reduce_sum", {{"axes", axes}}), inputs[0]);
                           return add_pointwise(
                               p2, rm, "main:pointwise0", {rs, inputs[1]}, single_pointwise("add"));
                       });
        auto sq = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {4, 5}}}), add);
        mm->add_return({sq});
    }
    EXPECT(p1.sort() == p2.sort());
}

// Same with a packed input: the unpack axis is a reduced axis, so it moves
// with the reduced axes and the packed bytes take the same reshape
TEST_CASE(unpack_reduce_reshape_transpose_pointwise)
{
    migraphx::shape ws{migraphx::shape::uint8_type, {1, 1, 8, 2, 2}};
    migraphx::shape ss{migraphx::shape::float_type, {1, 1, 8, 2, 4}};
    migraphx::shape ys{migraphx::shape::float_type, {1, 2, 1, 4}};
    auto dequant = [](auto* pm, const auto& xs) {
        auto c = pm->add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), xs[0]);
        return pm->add_instruction(migraphx::make_op("mul"), c, xs[1]);
    };
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto w    = mm->add_parameter("w", ws);
        auto s    = mm->add_parameter("s", ss);
        auto y    = mm->add_parameter("y", ys);
        auto rsum = add_reduce(
            p1,
            "main:reduce_sum0",
            {w, s},
            {3, 4},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto up =
                    rm->add_instruction(migraphx::make_op("unpack_int4", {{"axis", 4}}), inputs[0]);
                auto mul = add_pointwise(p1, rm, "main:pointwise0", {up, inputs[1]}, dequant);
                return rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
            });
        auto rsumr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 2, 4, 1, 1}}}), rsum);
        auto rsumt = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3, 4, 5}}}), rsumr);
        auto rsums = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {4, 5}}}), rsumt);
        auto add   = add_pointwise(p1, "main:pointwise1", {rsums, y}, single_pointwise("add"));
        mm->add_return({add});
    }
    run_pass(p1);

    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto w   = mm->add_parameter("w", ws);
        auto s   = mm->add_parameter("s", ss);
        auto y   = mm->add_parameter("y", ys);
        auto wr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 2, 4, 2, 2}}}), w);
        auto wt = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3, 4, 5}}}), wr);
        auto sr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {1, 1, 2, 4, 2, 4}}}), s);
        auto st = mm->add_instruction(
            migraphx::make_op("transpose", {{"permutation", {0, 2, 1, 3, 4, 5}}}), sr);
        auto yu  = mm->add_instruction(migraphx::make_op("unsqueeze", {{"axes", {4, 5}}}), y);
        auto add = add_reduce(
            p2,
            "main:reduce_sum0_reshape:main:pointwise1",
            {wt, st, yu},
            {4, 5},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto up =
                    rm->add_instruction(migraphx::make_op("unpack_int4", {{"axis", 5}}), inputs[0]);
                auto mul = add_pointwise(p2, rm, "main:pointwise0", {up, inputs[1]}, dequant);
                auto rs =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
                return add_pointwise(
                    p2, rm, "main:pointwise1", {rs, inputs[2]}, single_pointwise("add"));
            });
        auto sq = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {4, 5}}}), add);
        mm->add_return({sq});
    }
    EXPECT(p1.sort() == p2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
