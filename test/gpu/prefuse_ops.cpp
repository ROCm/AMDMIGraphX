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
#include <algorithm>
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

static bool runs_direct_conv(const migraphx::gpu::context& ctx)
{
    const auto gfx = ctx.get_current_device().get_gfx_name();
    return migraphx::starts_with(gfx, "gfx11") or migraphx::starts_with(gfx, "gfx12");
}

// enable_direct_conv offloads every dense convolution the kernel supports
TEST_CASE(direct_conv_enabled)
{
    migraphx::gpu::context ctx;
    if(not runs_direct_conv(ctx))
        test::skip("The direct conv kernel only runs on gfx11 and gfx12");
    migraphx::shape xs{migraphx::shape::float_type, {1, 8, 32, 32}};
    migraphx::shape ws{migraphx::shape::float_type, {16, 8, 5, 5}};
    migraphx::value attrs = {{"padding", {2, 2, 2, 2}}, {"stride", {2, 2}}, {"dilation", {1, 1}}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_parameter("w", ws);
        auto conv = m1.add_instruction(migraphx::make_op("convolution", attrs), x, w);
        m1.add_return({conv});
    }
    migraphx::run_passes(m1,
                         {migraphx::gpu::prefuse_ops{.ctx = &ctx, .enable_direct_conv = true},
                          migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", xs);
        auto w    = m2.add_parameter("w", ws);
        auto conv = m2.add_instruction(migraphx::make_op("gpu::direct_conv", attrs), x, w);
        m2.add_return({conv});
    }
    EXPECT(m1 == m2);
}

// disable_direct_conv wins over enable_direct_conv
TEST_CASE(direct_conv_disabled)
{
    migraphx::gpu::context ctx;
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 32, 32}});
        auto w    = m1.add_parameter("w", {migraphx::shape::float_type, {16, 8, 5, 5}});
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution", {{"padding", {2, 2}}, {"stride", {2, 2}}}), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    migraphx::run_passes(m1,
                         {migraphx::gpu::prefuse_ops{
                              .ctx = &ctx, .enable_direct_conv = true, .disable_direct_conv = true},
                          migraphx::dead_code_elimination{}});
    EXPECT(m1 == m2);
}

// Grouped convolutions are not dense, so the direct kernel leaves them
TEST_CASE(direct_conv_grouped)
{
    migraphx::gpu::context ctx;
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::float_type, {1, 8, 32, 32}});
        auto w    = m1.add_parameter("w", {migraphx::shape::float_type, {16, 4, 5, 5}});
        auto conv = m1.add_instruction(
            migraphx::make_op("convolution", {{"padding", {2, 2}}, {"group", 2}}), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    migraphx::run_passes(m1,
                         {migraphx::gpu::prefuse_ops{.ctx = &ctx, .enable_direct_conv = true},
                          migraphx::dead_code_elimination{}});
    EXPECT(m1 == m2);
}

// A filter of 33 rows needs more halo rows in registers than a lane keeps, even in the
// fallback solution
TEST_CASE(direct_conv_tall_filter)
{
    migraphx::gpu::context ctx;
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::float_type, {1, 4, 64, 64}});
        auto w    = m1.add_parameter("w", {migraphx::shape::float_type, {8, 4, 33, 1}});
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    migraphx::run_passes(m1,
                         {migraphx::gpu::prefuse_ops{.ctx = &ctx, .enable_direct_conv = true},
                          migraphx::dead_code_elimination{}});
    EXPECT(m1 == m2);
}

// Dilation 48 on a 3x3 filter needs (7 + 97) * (63 + 97) floats of halo for one input
// channel of the fallback 8x64 tile, more than the 64KB a gfx11 or gfx12 block may use
TEST_CASE(direct_conv_halo_too_large)
{
    migraphx::gpu::context ctx;
    migraphx::module m1;
    {
        auto x = m1.add_parameter("x", {migraphx::shape::float_type, {1, 4, 257, 257}});
        auto w = m1.add_parameter("w", {migraphx::shape::float_type, {8, 4, 3, 3}});
        auto conv =
            m1.add_instruction(migraphx::make_op("convolution", {{"dilation", {48, 48}}}), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    migraphx::run_passes(m1,
                         {migraphx::gpu::prefuse_ops{.ctx = &ctx, .enable_direct_conv = true},
                          migraphx::dead_code_elimination{}});
    EXPECT(m1 == m2);
}

// The largest dilation whose fallback halo still fits: (7 + 95) * (63 + 95) floats plus
// the filter is 64,500 bytes
TEST_CASE(direct_conv_halo_fits)
{
    migraphx::gpu::context ctx;
    if(not runs_direct_conv(ctx))
        test::skip("The direct conv kernel only runs on gfx11 and gfx12");
    migraphx::shape xs{migraphx::shape::float_type, {1, 4, 257, 257}};
    migraphx::shape ws{migraphx::shape::float_type, {8, 4, 3, 3}};
    migraphx::value attrs = {{"padding", {0, 0, 0, 0}}, {"stride", {1, 1}}, {"dilation", {47, 47}}};
    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", xs);
        auto w    = m1.add_parameter("w", ws);
        auto conv = m1.add_instruction(migraphx::make_op("convolution", attrs), x, w);
        m1.add_return({conv});
    }
    migraphx::run_passes(m1,
                         {migraphx::gpu::prefuse_ops{.ctx = &ctx, .enable_direct_conv = true},
                          migraphx::dead_code_elimination{}});

    migraphx::module m2;
    {
        auto x    = m2.add_parameter("x", xs);
        auto w    = m2.add_parameter("w", ws);
        auto conv = m2.add_instruction(migraphx::make_op("gpu::direct_conv", attrs), x, w);
        m2.add_return({conv});
    }
    EXPECT(m1 == m2);
}

// Runs prefuse_ops with its default heuristic and reports whether the convolution of x and
// w (with `attrs`) became a direct conv
static bool offloads_direct_conv(const migraphx::shape& xs,
                                 const migraphx::shape& ws,
                                 const migraphx::value& attrs)
{
    migraphx::gpu::context ctx;
    migraphx::module m;
    auto x = m.add_parameter("x", xs);
    auto w = m.add_parameter("w", ws);
    m.add_return({m.add_instruction(migraphx::make_op("convolution", attrs), x, w)});
    migraphx::run_passes(
        m, {migraphx::gpu::prefuse_ops{.ctx = &ctx}, migraphx::dead_code_elimination{}});
    return std::any_of(
        m.begin(), m.end(), [](const auto& ins) { return ins.name() == "gpu::direct_conv"; });
}

static migraphx::shape nhwc(migraphx::shape::type_t t, const std::vector<std::size_t>& lens)
{
    return migraphx::shape::from_permutation(t, lens, {0, 2, 3, 1});
}

// Few output channels over a large output: fp32 5x5 filters, fp32 channels-last 3x3 and
// strided filters, and fp16 filters of at least 25 taps in either layout, dilated or with
// up to 6 output channels for 7x7
TEST_CASE(direct_conv_heuristic_few_outputs)
{
    migraphx::gpu::context ctx;
    if(not migraphx::starts_with(ctx.get_current_device().get_gfx_name(), "gfx12"))
        test::skip("The direct conv heuristic only runs on gfx12");
    auto f32 = migraphx::shape::float_type;
    EXPECT(
        offloads_direct_conv({f32, {1, 8, 256, 256}}, {f32, {3, 8, 5, 5}}, {{"padding", {2, 2}}}));
    EXPECT(offloads_direct_conv(
        nhwc(f32, {1, 48, 256, 256}), nhwc(f32, {12, 48, 3, 3}), {{"padding", {1, 1}}}));
    EXPECT(offloads_direct_conv(nhwc(f32, {1, 24, 512, 512}),
                                nhwc(f32, {8, 24, 4, 4}),
                                {{"padding", {1, 1}}, {"stride", {2, 2}}}));
    auto f16 = migraphx::shape::half_type;
    EXPECT(offloads_direct_conv(
        {f16, {1, 48, 512, 512}}, {f16, {3, 48, 5, 5}}, {{"padding", {2, 2}}}));
    EXPECT(offloads_direct_conv(
        nhwc(f16, {1, 48, 512, 512}), nhwc(f16, {3, 48, 5, 5}), {{"padding", {2, 2}}}));
    EXPECT(offloads_direct_conv({f16, {1, 48, 512, 512}},
                                {f16, {3, 48, 5, 5}},
                                {{"padding", {4, 4}}, {"dilation", {2, 2}}}));
    EXPECT(offloads_direct_conv(
        nhwc(f16, {1, 32, 512, 512}), nhwc(f16, {6, 32, 7, 7}), {{"padding", {3, 3}}}));
}

// The GEMM-based kernels keep many output channels, small outputs, convs with little work
// per output, channels-first 3x3 filters and strides, 1x1 filters, 1-D convolutions, and
// fp16 filters under 25 taps, with a stride, or with 8 output channels
TEST_CASE(direct_conv_heuristic_default_lowering)
{
    auto f32 = migraphx::shape::float_type;
    EXPECT(not offloads_direct_conv(
        nhwc(f32, {1, 48, 256, 256}), nhwc(f32, {32, 48, 3, 3}), {{"padding", {1, 1}}}));
    EXPECT(not offloads_direct_conv(
        nhwc(f32, {1, 96, 128, 128}), nhwc(f32, {3, 96, 3, 3}), {{"padding", {1, 1}}}));
    EXPECT(not offloads_direct_conv(
        nhwc(f32, {1, 16, 512, 512}), nhwc(f32, {1, 16, 1, 1}), {{"padding", {0, 0}}}));
    EXPECT(not offloads_direct_conv(
        {f32, {1, 48, 256, 256}}, {f32, {3, 48, 3, 3}}, {{"padding", {1, 1}}}));
    EXPECT(not offloads_direct_conv(
        {f32, {1, 48, 512, 512}}, {f32, {3, 48, 5, 5}}, {{"padding", {2, 2}}, {"stride", {2, 2}}}));
    EXPECT(not offloads_direct_conv(
        nhwc(f32, {1, 64, 512, 512}), nhwc(f32, {3, 64, 1, 1}), {{"padding", {0, 0}}}));
    EXPECT(not offloads_direct_conv({f32, {1, 64, 65536}},
                                    {f32, {3, 64, 5}},
                                    {{"padding", {2}}, {"stride", {1}}, {"dilation", {1}}}));
    auto f16 = migraphx::shape::half_type;
    EXPECT(not offloads_direct_conv(
        nhwc(f16, {1, 48, 256, 256}), nhwc(f16, {3, 48, 3, 3}), {{"padding", {1, 1}}}));
    EXPECT(not offloads_direct_conv(
        {f16, {1, 48, 512, 512}}, {f16, {3, 48, 5, 5}}, {{"padding", {2, 2}}, {"stride", {2, 2}}}));
    EXPECT(not offloads_direct_conv(
        nhwc(f16, {1, 48, 512, 512}), nhwc(f16, {8, 48, 5, 5}), {{"padding", {2, 2}}}));
    EXPECT(not offloads_direct_conv(
        nhwc(f16, {1, 32, 512, 512}), nhwc(f16, {8, 32, 7, 7}), {{"padding", {3, 3}}}));
    EXPECT(not offloads_direct_conv(
        {f16, {1, 48, 128, 128}}, {f16, {3, 48, 5, 5}}, {{"padding", {2, 2}}}));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
