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

#include "verify_program.hpp"
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/generate.hpp>
#include <migraphx/common.hpp>
#include <migraphx/make_op.hpp>

// Regression for a GPU memory-access fault hit while benchmarking a fused
// GroupNorm(32) + SiLU reduce in the slme model (slme-v1-1fr-fp16-720x1280).
// The two-pass sqdiff variance keeps the reduces dependent so the op stays a
// single monolithic fused_reduce, whose block_size=1024 candidate compiled a
// register-array `block` kernel that spilled to scratch and page-faulted. The
// stats use reduce_sum + 1/N (not reduce_mean) to avoid rewrite_reduce folding
// them into the independent, splittable mean(x^2)-mean^2 form.
// See src/targets/gpu/jit/reduce.cpp (block vs block_large) and
// test/gpu/fused_reduce_block_size_guard.cpp (deterministic code-size guard).
template <migraphx::shape::type_t TYPE>
struct test_groupnorm_silu_fused_reduce : verify_program<test_groupnorm_silu_fused_reduce<TYPE>>
{
    migraphx::program create_program() const
    {
        migraphx::program p;
        auto* mm = p.get_main_module();

        const auto f = migraphx::shape::float_type;
        // 32 groups, per-group reduction 16*180*320 = 921600 (the slme block).
        const std::vector<std::size_t> nchw    = {1, 512, 180, 320};
        const std::vector<std::size_t> grouped = {1, 32, 921600};
        const std::vector<std::size_t> gstat   = {1, 32, 1};

        auto x = mm->add_parameter("x", migraphx::shape{TYPE, nchw});

        // Leading add so the normalized tensor is a computed value.
        auto conv_bias = mm->add_parameter("conv_bias", migraphx::shape{TYPE, {512}});
        auto conv_bias_b = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", nchw}}), conv_bias);
        auto added = mm->add_instruction(migraphx::make_op("add"), x, conv_bias_b);

        auto xf = mm->add_instruction(migraphx::make_op("convert", {{"target_type", f}}), added);
        auto xr = mm->add_instruction(migraphx::make_op("reshape", {{"dims", grouped}}), xf);

        // reduce_sum + 1/N (not reduce_mean) to avoid the sqdiff-variance rewrite.
        const float inv_n = 1.0f / 921600.0f;
        auto inv_n_lit    = mm->add_literal(migraphx::literal{migraphx::shape{f}, {inv_n}});
        auto inv_n_b =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", gstat}}), inv_n_lit);
        auto sum  = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), xr);
        auto mean = mm->add_instruction(migraphx::make_op("mul"), sum, inv_n_b);
        auto mean_b =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", grouped}}), mean);
        auto diff = mm->add_instruction(migraphx::make_op("sub"), xr, mean_b);

        // sqdiff variance keeps the second reduce dependent so it stays one kernel.
        auto mean_b2 =
            mm->add_instruction(migraphx::make_op("multibroadcast", {{"out_lens", grouped}}), mean);
        auto sqd     = mm->add_instruction(migraphx::make_op("sqdiff"), xr, mean_b2);
        auto var_sum = mm->add_instruction(migraphx::make_op("reduce_sum", {{"axes", {2}}}), sqd);
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
        auto gscale   = mm->add_parameter("gscale", migraphx::shape{f, {32}});
        auto gbias    = mm->add_parameter("gbias", migraphx::shape{f, {32}});
        auto gscale_b = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", grouped}}), gscale);
        auto gbias_b = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", grouped}}), gbias);
        auto gscaled = mm->add_instruction(migraphx::make_op("mul"), norm, gscale_b);
        auto gaffine = mm->add_instruction(migraphx::make_op("add"), gscaled, gbias_b);

        auto back = mm->add_instruction(migraphx::make_op("reshape", {{"dims", nchw}}), gaffine);

        // Per-channel affine (size 512).
        auto cscale   = mm->add_parameter("cscale", migraphx::shape{f, {512}});
        auto cbias    = mm->add_parameter("cbias", migraphx::shape{f, {512}});
        auto cscale_b = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", nchw}}), cscale);
        auto cbias_b = mm->add_instruction(
            migraphx::make_op("broadcast", {{"axis", 1}, {"out_lens", nchw}}), cbias);
        auto cscaled = mm->add_instruction(migraphx::make_op("mul"), back, cscale_b);
        auto caffine = mm->add_instruction(migraphx::make_op("add"), cscaled, cbias_b);

        // SiLU epilogue.
        auto sig  = mm->add_instruction(migraphx::make_op("sigmoid"), caffine);
        auto silu = mm->add_instruction(migraphx::make_op("mul"), caffine, sig);
        auto out  = mm->add_instruction(migraphx::make_op("convert", {{"target_type", TYPE}}), silu);
        mm->add_return({out});
        return p;
    }

    std::string section() const { return "reduce"; }
};

template struct test_groupnorm_silu_fused_reduce<migraphx::shape::half_type>;
