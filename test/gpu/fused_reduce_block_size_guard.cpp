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

// Deterministic guard (offline cross-compile, no GPU) for the memory-access
// fault the slme model hit while benchmarking a fused GroupNorm(32) + SiLU
// reduce. It cross-compiles the {block,256} and {block,1024} candidates for the
// faulting fused_reduce and asserts neither compiles the register-array `block`
// kernel that spills to scratch; benchmarking would otherwise keep only the
// fastest candidate and hide the oversized one.

#include <test.hpp>
#include <migraphx/program.hpp>
#include <migraphx/module.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/operation.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/pass.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/compile_options.hpp>
#include <migraphx/gpu/target.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/compiler.hpp>
#include <migraphx/gpu/device_description.hpp>
#include <migraphx/gpu/code_object_op.hpp>
#include <algorithm>
#include <iostream>

// slme GroupNorm(32) + SiLU block: dependent two-pass (sqdiff) variance keeps it
// a single monolithic fused_reduce. The 5D group layout {1,32,C/32,H,W} matches
// what the slme pipeline feeds the benchmarked reduce (a 3D reduce normalizes
// differently and misses the compiled candidate).
static migraphx::program make_groupnorm_silu_program(migraphx::shape::type_t type,
                                                     std::size_t channels,
                                                     std::size_t height,
                                                     std::size_t width)
{
    const std::size_t groups = 32;
    const std::size_t cpg    = channels / groups; // channels per group
    migraphx::program p;
    auto* mm = p.get_main_module();

    const auto f = migraphx::shape::float_type;
    const std::vector<std::size_t> nchw    = {1, channels, height, width};
    const std::vector<std::size_t> grouped = {1, groups, cpg, height, width};
    const std::vector<std::size_t> gstat   = {1, groups, 1, 1, 1};
    const std::vector<std::size_t> cstat   = {1, groups, cpg, 1, 1};
    const std::vector<int64_t> raxes       = {2, 3, 4};
    const float inv_n = 1.0f / static_cast<float>(cpg * height * width);

    auto x = mm->add_parameter("x", migraphx::shape{type, nchw});

    auto conv_bias   = mm->add_parameter("conv_bias", migraphx::shape{type, {channels}});
    auto conv_bias_b = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", nchw}}), conv_bias);
    auto added = mm->add_instruction(migraphx::make_op("add"), x, conv_bias_b);

    auto xf = mm->add_instruction(migraphx::make_op("convert", {{"target_type", f}}), added);
    auto xr = mm->add_instruction(migraphx::make_op("reshape", {{"dims", grouped}}), xf);

    auto inv_n_lit = mm->add_literal(migraphx::literal{migraphx::shape{f}, {inv_n}});
    auto inv_n_b   = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", gstat}}), inv_n_lit);
    auto sum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", raxes}}), xr);
    auto mean = mm->add_instruction(migraphx::make_op("mul"), sum, inv_n_b);
    auto mean_b =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", grouped}}), mean);
    auto diff = mm->add_instruction(migraphx::make_op("sub"), xr, mean_b);

    auto mean_b2 =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", grouped}}), mean);
    auto sqd     = mm->add_instruction(migraphx::make_op("sqdiff"), xr, mean_b2);
    auto var_sum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", raxes}}), sqd);
    auto var     = mm->add_instruction(migraphx::make_op("mul"), var_sum, inv_n_b);

    auto eps = mm->add_literal(migraphx::literal{migraphx::shape{f}, {1e-5f}});
    auto eps_b =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", gstat}}), eps);
    auto var_eps = mm->add_instruction(migraphx::make_op("add"), var, eps_b);
    auto rstd    = mm->add_instruction(migraphx::make_op("rsqrt"), var_eps);
    auto rstd_b =
        mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", grouped}}), rstd);
    auto norm = mm->add_instruction(migraphx::make_op("mul"), diff, rstd_b);

    // Per-group affine (size 32).
    auto gscale   = mm->add_parameter("gscale", migraphx::shape{f, {groups}});
    auto gbias    = mm->add_parameter("gbias", migraphx::shape{f, {groups}});
    auto gscale_b = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", grouped}}), gscale);
    auto gbias_b = mm->add_instruction(
        migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", grouped}}), gbias);
    auto gscaled = mm->add_instruction(migraphx::make_op("mul"), norm, gscale_b);
    auto gaffine = mm->add_instruction(migraphx::make_op("add"), gscaled, gbias_b);

    // Per-channel affine, stored as {1,32,cpg,1,1} like the slme literal.
    auto cscale   = mm->add_parameter("cscale", migraphx::shape{f, cstat});
    auto cbias    = mm->add_parameter("cbias", migraphx::shape{f, cstat});
    auto cscale_b = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", grouped}}), cscale);
    auto cbias_b = mm->add_instruction(
        migraphx::make_op("multibroadcast", {{"out_lens", grouped}}), cbias);
    auto cscaled = mm->add_instruction(migraphx::make_op("mul"), gaffine, cscale_b);
    auto caffine = mm->add_instruction(migraphx::make_op("add"), cscaled, cbias_b);

    auto sig  = mm->add_instruction(migraphx::make_op("sigmoid"), caffine);
    auto silu = mm->add_instruction(migraphx::make_op("mul"), caffine, sig);
    auto out  = mm->add_instruction(migraphx::make_op("convert", {{"target_type", type}}), silu);
    mm->add_return({out});
    return p;
}

static std::size_t compile_block_reduce(migraphx::gpu::context& ctx,
                                        migraphx::instruction_ref ins,
                                        const migraphx::operation& reduce_op,
                                        std::size_t block_size)
{
    migraphx::value solution =
        migraphx::value::object{{"algo", "block"}, {"block_size", block_size}};
    auto cr = migraphx::gpu::compile(ctx, ins, reduce_op, solution);
    EXPECT(cr.code_objects.size() == 1);
    return migraphx::any_cast<migraphx::gpu::code_object_op>(cr.code_objects.front())
        .code_object.size();
}

// Extract the operation a gpu::precompile_op wraps.
static migraphx::operation inner_op(migraphx::instruction_ref ins)
{
    return migraphx::from_value<migraphx::operation>(ins->get_operator().to_value().at("op"));
}

// block_large stays ~13KB; the slme `block` kernels were 143KB-907KB. 96KB
// cleanly separates them.
static constexpr std::size_t max_safe_code_object = 96 * 1024;

// Asserts every fused_reduce's {block,256} and {block,1024} candidates stay small.
static void check_no_block_reduce_explosion(std::size_t channels,
                                            std::size_t height,
                                            std::size_t width)
{
    auto p = make_groupnorm_silu_program(migraphx::shape::half_type, channels, height, width);

    // Cross-compile for gfx1100 (no physical device needed).
    migraphx::gpu::target t{migraphx::gpu::device_description{"gfx1100", 48, 1}};
    migraphx::compile_options options;
    auto gctx = t.get_context();

    // Run the real GPU pipeline up to (not including) compile_ops to get the
    // exact precompile_op(fused_reduce) the autotuner would benchmark.
    auto passes = t.get_passes(gctx, options);
    std::vector<migraphx::pass> prefix;
    for(const auto& ps : passes)
    {
        if(ps.name() == "gpu::compile_ops")
            break;
        prefix.push_back(ps);
    }
    migraphx::run_passes(p, prefix);

    auto& ctx = migraphx::any_cast<migraphx::gpu::context>(gctx);

    auto* mm = p.get_main_module();
    std::vector<migraphx::instruction_ref> freduces;
    for(auto ins = mm->begin(); ins != mm->end(); ++ins)
    {
        if(ins->name() != "gpu::precompile_op")
            continue;
        auto op_name = inner_op(ins).name();
        if(op_name == "fused_reduce" or op_name == "split_fused_reduce")
            freduces.push_back(ins);
    }
    EXPECT(not freduces.empty());

    for(auto ins : freduces)
    {
        auto reduce_op = inner_op(ins);
        for(std::size_t block_size : {std::size_t{256}, std::size_t{1024}})
        {
            auto size = compile_block_reduce(ctx, ins, reduce_op, block_size);
            std::cout << channels << "x" << height << "x" << width << " " << reduce_op
                      << ": block_size=" << block_size << " -> " << size << " bytes" << std::endl;
            EXPECT(size > 0);
            EXPECT(size < max_safe_code_object);
        }
    }
}

// slme group-norm layers whose `block` reduce kernels exploded (143KB-907KB) at
// block_size 256 and/or 1024 before the iteration-count fallback.
TEST_CASE(groupnorm_512c_180x320) { check_no_block_reduce_explosion(512, 180, 320); }
TEST_CASE(groupnorm_512c_90x160) { check_no_block_reduce_explosion(512, 90, 160); }
TEST_CASE(groupnorm_256c_180x320) { check_no_block_reduce_explosion(256, 180, 320); }

int main(int argc, const char* argv[]) { test::run(argc, argv); }
