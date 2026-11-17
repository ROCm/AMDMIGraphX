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

#include <migraphx/dead_code_elimination.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/prefuse_ops.hpp>
#include <migraphx/gpu/fuse_mlir.hpp>
#include <migraphx/gpu/gemm_softmax_gemm.hpp>
#include <migraphx/pass_manager.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/program.hpp>
#include <migraphx/generate.hpp>
#include <basic_ops.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/stringutils.hpp>
#include <test.hpp>

struct pre_gemm_softmax_gemm : migraphx::gpu::gemm_softmax_gemm
{
    std::string name() const { return "gpu::pre_gemm_softmax_gemm"; }
};

static void run_pass(migraphx::module& m)
{
    migraphx::run_passes(
        m,
        {migraphx::gpu::prefuse_ops{.enable_attention = true}, migraphx::dead_code_elimination{}});
}

TEST_CASE(find_gemm_softmax_gemm)
{
    migraphx::shape s1{migraphx::shape::float_type, {8, 16, 32}};
    migraphx::shape s2{migraphx::shape::float_type, {8, 32, 16}};

    migraphx::module m1;
    {
        auto x     = m1.add_parameter("x", s1);
        auto y     = m1.add_parameter("y", s2);
        auto z     = m1.add_parameter("z", s1);
        auto scale = m1.add_literal(2.0f);

        auto dot1     = m1.add_instruction(migraphx::make_op("dot"), x, y);
        auto scale_mb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", dot1->get_shape().lens()}}), scale);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), dot1, scale_mb);
        auto sm  = m1.add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), mul);
        m1.add_instruction(migraphx::make_op("dot"), sm, z);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s1);
        auto y = m2.add_parameter("y", s2);
        auto z = m2.add_parameter("z", s1);

        m2.add_instruction(pre_gemm_softmax_gemm{migraphx::make_op("dot"), 2}, x, y, z);
    }

    EXPECT(m1 == m2);
}

TEST_CASE(find_gemm_softmax_gemm_multi_scale)
{
    migraphx::shape s1{migraphx::shape::float_type, {8, 16, 32}};
    migraphx::shape s2{migraphx::shape::float_type, {8, 32, 16}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s1);
        auto y = m1.add_parameter("y", s2);
        auto z = m1.add_parameter("z", s1);
        auto scale =
            m1.add_literal(migraphx::generate_literal({migraphx::shape::float_type, {16}}, 10));

        auto dot1     = m1.add_instruction(migraphx::make_op("dot"), x, y);
        auto scale_mb = m1.add_instruction(
            migraphx::make_op("multibroadcast", {{"out_lens", dot1->get_shape().lens()}}), scale);
        auto mul = m1.add_instruction(migraphx::make_op("mul"), dot1, scale_mb);
        auto sm  = m1.add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), mul);
        m1.add_instruction(migraphx::make_op("dot"), sm, z);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s1);
        auto y = m2.add_parameter("y", s2);
        auto z = m2.add_parameter("z", s1);

        m2.add_instruction(pre_gemm_softmax_gemm{migraphx::make_op("dot"), 1}, x, y, z);
    }

    EXPECT(m1 == m2);
}

TEST_CASE(find_gemm_softmax_gemm_no_scale)
{
    migraphx::shape s1{migraphx::shape::float_type, {8, 16, 32}};
    migraphx::shape s2{migraphx::shape::float_type, {8, 32, 16}};

    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", s1);
        auto y = m1.add_parameter("y", s2);
        auto z = m1.add_parameter("z", s1);

        auto dot1 = m1.add_instruction(migraphx::make_op("dot"), x, y);
        auto sm   = m1.add_instruction(migraphx::make_op("softmax", {{"axis", 2}}), dot1);
        m1.add_instruction(migraphx::make_op("dot"), sm, z);
    }
    run_pass(m1);

    migraphx::module m2;
    {
        auto x = m2.add_parameter("x", s1);
        auto y = m2.add_parameter("y", s2);
        auto z = m2.add_parameter("z", s1);

        m2.add_instruction(pre_gemm_softmax_gemm{migraphx::make_op("dot"), 1}, x, y, z);
    }

    EXPECT(m1 == m2);
}

// A depthwise convolution whose halo fits in shared memory becomes a channelwise conv on
// the gfx11 and gfx12 devices that run the channelwise kernel
TEST_CASE(channelwise_conv_halo_fits)
{
    migraphx::gpu::context ctx;
    const auto gfx = ctx.get_current_device().get_gfx_name();
    if(not migraphx::starts_with(gfx, "gfx11") and not migraphx::starts_with(gfx, "gfx12"))
        test::skip("The channelwise kernel only runs on gfx11 and gfx12");
    migraphx::shape xs{migraphx::shape::float_type, {1, 4, 64, 64}};
    migraphx::shape ws{migraphx::shape::float_type, {4, 1, 3, 3}};
    migraphx::value attrs = {{"padding", {2, 2, 2, 2}}, {"stride", {1, 1}}, {"dilation", {2, 2}}};
    migraphx::module m1;
    {
        auto x          = m1.add_parameter("x", xs);
        auto w          = m1.add_parameter("w", ws);
        auto conv_v     = attrs;
        conv_v["group"] = 4;
        auto conv       = m1.add_instruction(migraphx::make_op("convolution", conv_v), x, w);
        m1.add_return({conv});
    }
    migraphx::run_passes(
        m1, {migraphx::gpu::prefuse_ops{.ctx = &ctx}, migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", xs);
        auto w    = m2.add_parameter("w", ws);
        auto conv = m2.add_instruction(migraphx::make_op("gpu::channelwise_conv", attrs), x, w);
        m2.add_return({conv});
    }
    EXPECT(m1 == m2);
}

// Dilation 64 needs 66,564 bytes of halo for a single float output, more than a gfx11 or
// gfx12 block may use, so the convolution is left for the other backends
TEST_CASE(channelwise_conv_halo_too_large)
{
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::float_type, {1, 4, 257, 257}});
        auto w    = m1.add_parameter("w", {migraphx::shape::float_type, {4, 1, 3, 3}});
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution", {{"group", 4}, {"dilation", {64, 64}}}), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    migraphx::gpu::context ctx;
    migraphx::run_passes(
        m1, {migraphx::gpu::prefuse_ops{.ctx = &ctx}, migraphx::dead_code_elimination{}});
    EXPECT(m1 == m2);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
