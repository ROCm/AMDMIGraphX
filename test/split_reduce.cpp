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
#include <migraphx/split_reduce.hpp>
#include <migraphx/fuse_pointwise.hpp>
#include <migraphx/fuse_reduce.hpp>
#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/program.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>

#include <test.hpp>
#include <pointwise.hpp>
#include <reduce.hpp>

static void run_pass(migraphx::program& p, const migraphx::split_reduce& sr)
{
    migraphx::run_passes(p,
                         {migraphx::fuse_pointwise{},
                          migraphx::fuse_reduce{},
                          sr,
                          migraphx::fuse_pointwise{.enable_rewrite_broadcasts = true},
                          migraphx::dead_code_elimination{}});
}

static void run_pass(migraphx::program& p) { run_pass(p, {.split_size = 8192}); }

static void run_fuse_pass(migraphx::program& p)
{
    migraphx::run_passes(
        p,
        {migraphx::fuse_pointwise{}, migraphx::fuse_reduce{}, migraphx::dead_code_elimination{}});
}

TEST_CASE(single)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        mm->add_return({rsum});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto xr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 64, 5120}}}), x);
        auto partial =
            add_reduce(p2, "main:reduce_sum0_split", {xr}, {3}, single_reduce("reduce_sum"));
        auto sq = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), partial);
        auto rsum =
            add_reduce(p2, "main:reduce_sum0_final", {sq}, {2}, single_reduce("reduce_sum"));
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(single_reduce_max)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rmax = mm->add_instruction(migraphx::make_op("reduce_max", {{"axes", {2}}}), x);
        mm->add_return({rmax});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto xr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 64, 5120}}}), x);
        auto partial =
            add_reduce(p2, "main:reduce_max0_split", {xr}, {3}, single_reduce("reduce_max"));
        auto sq = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), partial);
        auto rmax =
            add_reduce(p2, "main:reduce_max0_final", {sq}, {2}, single_reduce("reduce_max"));
        mm->add_return({rmax});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(multi_axis)
{
    migraphx::shape s{migraphx::shape::float_type, {14400, 4, 20, 4}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 2}}}), x);
        mm->add_return({rsum});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto xr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {64, 225, 4, 20, 4}}}), x);
        auto partial =
            add_reduce(p2, "main:reduce_sum0_split", {xr}, {1, 3}, single_reduce("reduce_sum"));
        auto sq = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), partial);
        auto rsum =
            add_reduce(p2, "main:reduce_sum0_final", {sq}, {0, 2}, single_reduce("reduce_sum"));
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(many_outputs)
{
    // With enough outputs the reduction already has enough parallelism, so
    // a reduction below the split_size is not split at all
    migraphx::shape s{migraphx::shape::float_type, {14400, 32, 20, 16}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 2}}}), x);
        mm->add_return({rsum});
    }
    migraphx::program p2 = p1;
    run_fuse_pass(p2);
    run_pass(p1, {.split_size = 1048576, .partial_split_size = 8192});
    EXPECT(p1 == p2);
}

TEST_CASE(many_outputs_large)
{
    // Beyond the split_size the reduction is too large for a single
    // workgroup, so it is split even with many outputs
    migraphx::shape s{migraphx::shape::float_type, {14400, 32, 20, 16}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {0, 2}}}), x);
        mm->add_return({rsum});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto xr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {64, 225, 32, 20, 16}}}), x);
        auto partial =
            add_reduce(p2, "main:reduce_sum0_split", {xr}, {1, 3}, single_reduce("reduce_sum"));
        auto sq = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {1}}}), partial);
        auto rsum =
            add_reduce(p2, "main:reduce_sum0_final", {sq}, {0, 2}, single_reduce("reduce_sum"));
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(prefer_atomic)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        mm->add_return({rsum});
    }
    run_pass(p1, {.split_size = 8192, .prefer_partial_reduce = false});
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p2, "main:reduce_sum0_split", {x}, {2}, "assign_add", single_reduce("reduce_sum"));
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(partial_threshold_only)
{
    // The atomic threshold is not met, so the partial reduce is used even
    // though prefer_partial_reduce is false
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        mm->add_return({rsum});
    }
    run_pass(p1,
             {.split_size = 1048576, .partial_split_size = 8192, .prefer_partial_reduce = false});
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto xr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 64, 5120}}}), x);
        auto partial =
            add_reduce(p2, "main:reduce_sum0_split", {xr}, {3}, single_reduce("reduce_sum"));
        auto sq = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), partial);
        auto rsum =
            add_reduce(p2, "main:reduce_sum0_final", {sq}, {2}, single_reduce("reduce_sum"));
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(atomic_threshold_only)
{
    // The partial threshold is not met, so the atomic split is used even
    // though prefer_partial_reduce is true
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        mm->add_return({rsum});
    }
    run_pass(p1, {.split_size = 8192, .partial_split_size = 1048576});
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p2, "main:reduce_sum0_split", {x}, {2}, "assign_add", single_reduce("reduce_sum"));
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(atomic_fallback)
{
    // 13117 = 13 * 1009 cant be split into groups, so the atomic-based
    // split_fused_reduce is used instead
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 13117}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        mm->add_return({rsum});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p2, "main:reduce_sum0_split", {x}, {2}, "assign_add", single_reduce("reduce_sum"));
        mm->add_return({rsum});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(fused)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto add = mm->add_instruction(migraphx::make_op("add"), x, rsumb);
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto xr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 64, 5120}}}), x);
        auto partial = add_reduce(
            p2, "main:reduce_sum0:main:pointwise0_split", {xr}, {3}, single_reduce("reduce_sum"));
        auto sq   = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), partial);
        auto rsum = add_reduce(
            p2, "main:reduce_sum0:main:pointwise0_final", {sq}, {2}, single_reduce("reduce_sum"));
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto add = add_pointwise(p2, mm, "main:pointwise0", {x, rsumb}, single_pointwise("add"));
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(fused_trailing)
{
    // With enough outputs the trailing operators are fused into the
    // completion kernel instead of being inserted into the parent module
    migraphx::shape s{migraphx::shape::float_type, {4, 4, 327680}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto add = mm->add_instruction(migraphx::make_op("add"), x, rsumb);
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto xr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {4, 4, 64, 5120}}}), x);
        auto partial = add_reduce(
            p2, "main:reduce_sum0:main:pointwise0_split", {xr}, {3}, single_reduce("reduce_sum"));
        auto sq  = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), partial);
        auto add = add_reduce(
            p2,
            "main:reduce_sum0:main:pointwise0_final",
            {sq, x},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum  = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
                return add_pointwise(
                    p2, rm, "main:pointwise0", {inputs[1], rsumb}, single_pointwise("add"));
            });
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(small)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 1024}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto add = mm->add_instruction(migraphx::make_op("add"), x, rsumb);
        mm->add_return({add});
    }
    migraphx::program p2 = p1;
    run_fuse_pass(p2);
    run_pass(p1);

    EXPECT(p1 == p2);
}

TEST_CASE(split_pointwise)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto sqrt  = mm->add_instruction(migraphx::make_op("sqrt"), x);
        auto rsum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), sqrt);
        auto rsumb = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto add = mm->add_instruction(migraphx::make_op("add"), sqrt, rsumb);
        mm->add_return({add});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm  = p2.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto sqrt = add_pointwise(p2, mm, "main:pointwise0", {x}, single_pointwise("sqrt"));
        auto xr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 64, 5120}}}), sqrt);
        auto partial = add_reduce(p2,
                                  "main:pointwise0:main:reduce_sum0:main:pointwise1_split",
                                  {xr},
                                  {3},
                                  single_reduce("reduce_sum"));
        auto sq      = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), partial);
        auto rsum    = add_reduce(p2,
                                  "main:pointwise0:main:reduce_sum0:main:pointwise1_final",
                                  {sq},
                                  {2},
                                  single_reduce("reduce_sum"));
        auto rsumb   = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum);
        auto add = add_pointwise(p2, mm, "main:pointwise1", {sqrt, rsumb}, single_pointwise("add"));
        mm->add_return({add});
    }
    EXPECT(p1 == p2);
}

TEST_CASE(sequence_reduces)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm    = p1.get_main_module();
        auto x      = mm->add_parameter("x", s);
        auto rsum1  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsum1b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum1);
        auto sub    = mm->add_instruction(migraphx::make_op("sub"), x, rsum1b);
        auto rsum2  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), sub);
        auto rsum2b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), rsum2);
        auto add = mm->add_instruction(migraphx::make_op("add"), rsum2b, x);
        mm->add_return({add});
    }
    migraphx::program p2 = p1;
    run_fuse_pass(p2);
    run_pass(p1);

    EXPECT(p1 == p2);
}

TEST_CASE(parallel_reduce)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm   = p1.get_main_module();
        auto x     = mm->add_parameter("x", s);
        auto xx    = mm->add_instruction(migraphx::make_op("mul"), x, x);
        auto rsum1 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), x);
        auto rsum2 = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), xx);
        auto mul   = mm->add_instruction(migraphx::make_op("mul"), rsum1, rsum2);
        mm->add_return({mul});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto xr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 64, 5120}}}), x);
        auto partials = add_reduce(
            p2,
            "main:reduce_sum0:main:pointwise1:main:pointwise0:main:reduce_sum1_split",
            {xr},
            {3},
            [&](auto* rm,
                const auto& inputs,
                const auto& axes) -> std::vector<migraphx::instruction_ref> {
                auto xx = add_pointwise(p2, rm, "main:pointwise0", {inputs[0]}, squared());
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), xx);
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                return {rsum2, rsum1};
            });
        auto rsum2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), partials);
        auto sq2 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), rsum2);
        auto rsum1 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), partials);
        auto sq1 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), rsum1);
        auto mul = add_reduce(
            p2,
            "main:reduce_sum0:main:pointwise1:main:pointwise0:main:reduce_sum1_final",
            {sq2, sq1},
            {2},
            [&](auto* rm, const auto& inputs, const auto& axes) {
                auto crsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                  inputs[1]);
                auto crsum2 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                  inputs[0]);
                return add_pointwise(
                    p2, rm, "main:pointwise1", {crsum1, crsum2}, single_pointwise("mul"));
            });
        mm->add_return({mul});
    }
    EXPECT(p1.sort() == p2.sort());
}

TEST_CASE(double_split_live)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p1, "fuse_reduce0", {x}, {2}, [&](auto* rm, const auto& inputs, const auto& axes) {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto sqrt =
                    add_pointwise(p1, rm, "main:pointwise0", {rsum1}, single_pointwise("sqrt"));
                auto sqrtb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), sqrt);
                auto mul = add_pointwise(p1, rm, "main:pointwise1", {inputs[0]}, squared());
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
                auto add = add_pointwise(
                    p1, rm, "main:pointwise2", {rsum2, sqrt}, single_pointwise("add"));
                auto addb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), add);
                return add_pointwise(
                    p1, rm, "main:pointwise3", {addb, sqrtb}, single_pointwise("mul"));
            });
        mm->add_return({rsum});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        auto xr =
            mm->add_instruction(migraphx::make_op("reshape", {{"dims", {2, 3, 64, 5120}}}), x);
        auto partials = add_reduce(
            p2,
            "fuse_reduce0_split",
            {xr},
            {3},
            [&](auto* rm,
                const auto& inputs,
                const auto& axes) -> std::vector<migraphx::instruction_ref> {
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto mul   = add_pointwise(p2, rm, "main:pointwise1", {inputs[0]}, squared());
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), mul);
                return {rsum1, rsum2};
            });
        auto rsum1 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), partials);
        auto sq1 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), rsum1);
        auto rsum2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), partials);
        auto sq2 = mm->add_instruction(migraphx::make_op("squeeze", {{"axes", {3}}}), rsum2);
        auto completed =
            add_reduce(p2,
                       "fuse_reduce0_final",
                       {sq1, sq2},
                       {2},
                       [&](auto* rm,
                           const auto& inputs,
                           const auto& axes) -> std::vector<migraphx::instruction_ref> {
                           auto crsum1 = rm->add_instruction(
                               migraphx::make_op("reduce_sum", {{"axes", axes}}), inputs[0]);
                           auto crsum2 = rm->add_instruction(
                               migraphx::make_op("reduce_sum", {{"axes", axes}}), inputs[1]);
                           return {crsum1, crsum2};
                       });
        auto crsum1 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), completed);
        auto crsum1b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), crsum1);
        auto crsum2 =
            mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), completed);
        auto crsum2b = mm->add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), crsum2);
        auto sqrt_add_mul = add_pointwise(
            p2, "main:pointwise0", {crsum1b, crsum2b}, [](auto* pm, const auto& inputs) {
                auto sqrt = pm->add_instruction(migraphx::make_op("sqrt"), inputs[0]);
                auto add  = pm->add_instruction(migraphx::make_op("add"), inputs[1], sqrt);
                return pm->add_instruction(migraphx::make_op("mul"), add, sqrt);
            });
        mm->add_return({sqrt_add_mul});
    }
    EXPECT(p1.sort() == p2.sort());
}

// Test multi-alias in parallel reduce scenario - both reduce outputs are aliased by multi_alias_op
// The pass should split both reduces and extract the multi_alias to the main module
TEST_CASE(parallel_reduce_multi_alias)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p1, "fuse_reduce0", {x}, {2}, [&](auto* rm, const auto& inputs, const auto& axes) {
                auto xx    = add_pointwise(p1, rm, "main:pointwise0", {inputs[0]}, squared());
                auto rsum1 = rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}),
                                                 inputs[0]);
                auto rsum2 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), xx);
                // multi_alias_op aliases both reduce outputs
                return rm->add_instruction(multi_alias_op{}, rsum1, rsum2);
            });
        mm->add_return({rsum});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        // The pointwise (squared) is extracted to main module
        auto xx = add_pointwise(p2, mm, "main:pointwise0", {x}, squared());
        // Split module takes both xx and x as inputs
        auto rsum =
            add_reduce(p2,
                       "fuse_reduce0_split",
                       {xx, x},
                       {2},
                       "assign_add",
                       [&](auto* rm,
                           const auto& inputs,
                           const auto& axes) -> std::vector<migraphx::instruction_ref> {
                           // inputs[0] is xx (squared), inputs[1] is x
                           // The pass returns (rsum1, rsum2) order based on the original fused
                           // module order
                           auto rsum1 = rm->add_instruction(
                               migraphx::make_op("reduce_sum", {{"axes", axes}}), inputs[1]);
                           auto rsum2 = rm->add_instruction(
                               migraphx::make_op("reduce_sum", {{"axes", axes}}), inputs[0]);
                           return {rsum1, rsum2};
                       });
        auto rsum1 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 0}}), rsum);
        auto rsum2 = mm->add_instruction(migraphx::make_op("get_tuple_elem", {{"index", 1}}), rsum);
        // multi_alias_op is moved to main module after split
        auto ma = mm->add_instruction(multi_alias_op{}, rsum1, rsum2);
        mm->add_return({ma});
    }
    EXPECT(p1.sort() == p2.sort());
}

// Test that find_alive correctly identifies live instructions through multi-alias chain
// sqrt is computed before reduce, used after reduce through multi_alias - should be split out
TEST_CASE(split_with_multi_alias_alive)
{
    migraphx::shape s{migraphx::shape::float_type, {2, 3, 327680}};
    migraphx::program p1;
    {
        auto* mm  = p1.get_main_module();
        auto x    = mm->add_parameter("x", s);
        auto rsum = add_reduce(
            p1, "fuse_reduce0", {x}, {2}, [&](auto* rm, const auto& inputs, const auto& axes) {
                // Create a computation before the reduce
                auto sqrt =
                    add_pointwise(p1, rm, "main:pointwise0", {inputs[0]}, single_pointwise("sqrt"));
                auto rsum1 =
                    rm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", axes}}), sqrt);
                // multi_alias aliases sqrt and rsum1 - sqrt should be identified as alive
                auto ma    = rm->add_instruction(multi_alias_op{}, sqrt, rsum1);
                auto rsumb = rm->add_instruction(
                    migraphx::make_op("multibroadcast", {{"out_lens", s.lens()}}), ma);
                return add_pointwise(
                    p1, rm, "main:pointwise1", {rsumb, sqrt}, single_pointwise("mul"));
            });
        mm->add_return({rsum});
    }
    run_pass(p1);
    migraphx::program p2;
    {
        auto* mm = p2.get_main_module();
        auto x   = mm->add_parameter("x", s);
        // sqrt is computed first, then passed to split module
        auto sqrt = add_pointwise(p2, mm, "main:pointwise0", {x}, single_pointwise("sqrt"));
        auto rsums =
            add_reduce(p2,
                       "fuse_reduce0_split",
                       {sqrt},
                       {2},
                       "assign_add",
                       [&](auto* rm, const auto& inputs, const auto& axes) {
                           return rm->add_instruction(
                               migraphx::make_op("reduce_sum", {{"axes", axes}}), inputs[0]);
                       });
        // After split: multi_alias(sqrt, rsums) - shape is {2,3,327680} from sqrt
        // multibroadcast is eliminated since multi_alias already has the right shape
        auto ma = mm->add_instruction(multi_alias_op{}, sqrt, rsums);
        // multiply multi_alias result with sqrt
        auto result = add_pointwise(p2, mm, "main:pointwise1", {ma, sqrt}, single_pointwise("mul"));
        mm->add_return({result});
    }
    EXPECT(p1.sort() == p2.sort());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
