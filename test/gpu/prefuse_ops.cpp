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
#include <migraphx/gpu/device_description.hpp>
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

// enable_direct_conv offloads every dense convolution the kernel supports
TEST_CASE(direct_conv_enabled)
{
    migraphx::gpu::context ctx;
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
// channel of the fallback 8x64 tile, more than the 64KB of shared memory a block may use
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
    if(ctx.get_current_device().get_max_shared_memory_per_block() < 64 * 1024)
        test::skip("The halo needs 64KB of shared memory per block");
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

// Runs prefuse_ops with its default heuristic for `device` and reports whether the
// convolution of x and w (with `attrs`) became a direct conv
static bool offloads_direct_conv(const migraphx::gpu::device_description& device,
                                 const migraphx::shape& xs,
                                 const migraphx::shape& ws,
                                 const migraphx::value& attrs)
{
    migraphx::gpu::context ctx{device};
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

// 32 wave32 workgroup processors, which are 64 compute units: fp32 has no matrix
// instructions and fp16 has WMMA
static migraphx::gpu::device_description gfx1201() { return {"gfx1201", 32}; }

// 304 compute units with MFMA for fp32 and fp16
static migraphx::gpu::device_description gfx942() { return {"gfx942", 304}; }

// Without matrix instructions: fp32 5x5 filters, fp32 channels-last 3x3 filters and strided
// 4x4 filters with up to 16 output channels. With WMMA: fp16 filters of at least 25 taps in
// either layout, dilated or with up to 6 output channels for 7x7
TEST_CASE(direct_conv_heuristic_wave32_few_outputs)
{
    auto f32 = migraphx::shape::float_type;
    EXPECT(offloads_direct_conv(
        gfx1201(), {f32, {1, 8, 256, 256}}, {f32, {3, 8, 5, 5}}, {{"padding", {2, 2}}}));
    EXPECT(offloads_direct_conv(
        gfx1201(), nhwc(f32, {1, 48, 256, 256}), nhwc(f32, {12, 48, 3, 3}), {{"padding", {1, 1}}}));
    EXPECT(offloads_direct_conv(gfx1201(),
                                nhwc(f32, {1, 24, 512, 512}),
                                nhwc(f32, {8, 24, 4, 4}),
                                {{"padding", {1, 1}}, {"stride", {2, 2}}}));
    auto f16 = migraphx::shape::half_type;
    EXPECT(offloads_direct_conv(
        gfx1201(), {f16, {1, 48, 512, 512}}, {f16, {3, 48, 5, 5}}, {{"padding", {2, 2}}}));
    EXPECT(offloads_direct_conv(
        gfx1201(), nhwc(f16, {1, 48, 512, 512}), nhwc(f16, {3, 48, 5, 5}), {{"padding", {2, 2}}}));
    EXPECT(offloads_direct_conv(gfx1201(),
                                {f16, {1, 48, 512, 512}},
                                {f16, {3, 48, 5, 5}},
                                {{"padding", {4, 4}}, {"dilation", {2, 2}}}));
    EXPECT(offloads_direct_conv(
        gfx1201(), nhwc(f16, {1, 32, 512, 512}), nhwc(f16, {6, 32, 7, 7}), {{"padding", {3, 3}}}));
}

// The GEMM-based kernels keep many output channels, small outputs, convs with little work
// per output, channels-first 3x3 filters and strides, channels-last 3x3 strides, 1x1
// filters, 1-D convolutions, and fp16 filters under 25 taps, with a stride, or with 8
// output channels. MFMA devices leave all of these too.
TEST_CASE(direct_conv_heuristic_default_lowering)
{
    auto offloads =
        [](const migraphx::shape& xs, const migraphx::shape& ws, const migraphx::value& attrs) {
            return offloads_direct_conv(gfx1201(), xs, ws, attrs) or
                   offloads_direct_conv(gfx942(), xs, ws, attrs);
        };
    auto f32 = migraphx::shape::float_type;
    EXPECT(not offloads(
        nhwc(f32, {1, 48, 256, 256}), nhwc(f32, {32, 48, 3, 3}), {{"padding", {1, 1}}}));
    EXPECT(not offloads(
        nhwc(f32, {1, 96, 128, 128}), nhwc(f32, {3, 96, 3, 3}), {{"padding", {1, 1}}}));
    EXPECT(not offloads(
        nhwc(f32, {1, 16, 512, 512}), nhwc(f32, {1, 16, 1, 1}), {{"padding", {0, 0}}}));
    EXPECT(not offloads({f32, {1, 48, 256, 256}}, {f32, {3, 48, 3, 3}}, {{"padding", {1, 1}}}));
    EXPECT(not offloads(
        {f32, {1, 48, 512, 512}}, {f32, {3, 48, 5, 5}}, {{"padding", {2, 2}}, {"stride", {2, 2}}}));
    EXPECT(not offloads(nhwc(f32, {1, 64, 640, 640}),
                        nhwc(f32, {3, 64, 3, 3}),
                        {{"padding", {1, 1}}, {"stride", {2, 2}}}));
    EXPECT(not offloads(
        nhwc(f32, {1, 64, 512, 512}), nhwc(f32, {3, 64, 1, 1}), {{"padding", {0, 0}}}));
    EXPECT(not offloads({f32, {1, 64, 65536}},
                        {f32, {3, 64, 5}},
                        {{"padding", {2}}, {"stride", {1}}, {"dilation", {1}}}));
    auto f16 = migraphx::shape::half_type;
    EXPECT(not offloads(
        nhwc(f16, {1, 48, 256, 256}), nhwc(f16, {3, 48, 3, 3}), {{"padding", {1, 1}}}));
    EXPECT(not offloads(
        {f16, {1, 48, 512, 512}}, {f16, {3, 48, 5, 5}}, {{"padding", {2, 2}}, {"stride", {2, 2}}}));
    EXPECT(not offloads(
        nhwc(f16, {1, 48, 512, 512}), nhwc(f16, {8, 48, 5, 5}), {{"padding", {2, 2}}}));
    EXPECT(not offloads(
        nhwc(f16, {1, 32, 512, 512}), nhwc(f16, {8, 32, 7, 7}), {{"padding", {3, 3}}}));
    EXPECT(not offloads({f16, {1, 48, 128, 128}}, {f16, {3, 48, 5, 5}}, {{"padding", {2, 2}}}));
}

// With MFMA: fp32 with up to 3 output channels, or 6 for 7x7 filters; channels-first fp32
// dilated filters; channels-last fp16 with up to 3 output channels over few input channels;
// channels-first fp16 large or dilated filters
TEST_CASE(direct_conv_heuristic_mfma_few_outputs)
{
    auto f32 = migraphx::shape::float_type;
    EXPECT(offloads_direct_conv(
        gfx942(), nhwc(f32, {1, 64, 640, 640}), nhwc(f32, {3, 64, 3, 3}), {{"padding", {1, 1}}}));
    EXPECT(offloads_direct_conv(
        gfx942(), nhwc(f32, {1, 64, 256, 256}), nhwc(f32, {3, 64, 5, 5}), {{"padding", {2, 2}}}));
    EXPECT(offloads_direct_conv(
        gfx942(), nhwc(f32, {1, 64, 640, 640}), nhwc(f32, {6, 64, 7, 7}), {{"padding", {3, 3}}}));
    EXPECT(offloads_direct_conv(gfx942(),
                                {f32, {1, 64, 256, 256}},
                                {f32, {8, 64, 3, 3}},
                                {{"padding", {2, 2}}, {"dilation", {2, 2}}}));
    auto f16 = migraphx::shape::half_type;
    EXPECT(offloads_direct_conv(
        gfx942(), nhwc(f16, {1, 16, 640, 640}), nhwc(f16, {1, 16, 3, 3}), {{"padding", {1, 1}}}));
    EXPECT(offloads_direct_conv(
        gfx942(), nhwc(f16, {1, 64, 640, 640}), nhwc(f16, {1, 64, 7, 7}), {{"padding", {3, 3}}}));
    EXPECT(offloads_direct_conv(
        gfx942(), {f16, {1, 64, 640, 640}}, {f16, {8, 64, 7, 7}}, {{"padding", {3, 3}}}));
    EXPECT(offloads_direct_conv(gfx942(),
                                {f16, {1, 16, 384, 384}},
                                {f16, {3, 16, 3, 3}},
                                {{"padding", {2, 2}}, {"dilation", {2, 2}}}));
}

// With MFMA, MLIR keeps fp32 with 6 output channels under 49 taps, channels-last fp32
// dilated filters, strides, channels-last fp16 over few output positions or with large
// filters over many input channels, and channels-first fp16 3x3 filters
TEST_CASE(direct_conv_heuristic_mfma_default_lowering)
{
    auto f32 = migraphx::shape::float_type;
    EXPECT(not offloads_direct_conv(
        gfx942(), nhwc(f32, {1, 64, 640, 640}), nhwc(f32, {6, 64, 5, 5}), {{"padding", {2, 2}}}));
    EXPECT(not offloads_direct_conv(
        gfx942(), {f32, {1, 3, 128, 128}}, {f32, {64, 3, 3, 3}}, {{"padding", {1, 1}}}));
    EXPECT(not offloads_direct_conv(gfx942(),
                                    nhwc(f32, {1, 64, 640, 640}),
                                    nhwc(f32, {3, 64, 3, 3}),
                                    {{"padding", {2, 2}}, {"dilation", {2, 2}}}));
    EXPECT(not offloads_direct_conv(gfx942(),
                                    nhwc(f32, {1, 64, 768, 768}),
                                    nhwc(f32, {1, 64, 3, 3}),
                                    {{"padding", {1, 1}}, {"stride", {2, 2}}}));
    auto f16 = migraphx::shape::half_type;
    EXPECT(not offloads_direct_conv(
        gfx942(), nhwc(f16, {1, 16, 384, 384}), nhwc(f16, {3, 16, 7, 7}), {{"padding", {3, 3}}}));
    EXPECT(not offloads_direct_conv(
        gfx942(), nhwc(f16, {1, 64, 640, 640}), nhwc(f16, {3, 64, 5, 5}), {{"padding", {2, 2}}}));
    EXPECT(not offloads_direct_conv(
        gfx942(), nhwc(f16, {1, 48, 512, 512}), nhwc(f16, {3, 48, 5, 5}), {{"padding", {2, 2}}}));
    EXPECT(not offloads_direct_conv(
        gfx942(), {f16, {1, 64, 640, 640}}, {f16, {3, 64, 3, 3}}, {{"padding", {1, 1}}}));
    EXPECT(not offloads_direct_conv(gfx942(),
                                    {f16, {1, 16, 1280, 1280}},
                                    {f16, {1, 16, 5, 5}},
                                    {{"padding", {2, 2}}, {"stride", {2, 2}}}));
}

// The output positions a convolution needs grow with the compute units it has to fill
TEST_CASE(direct_conv_heuristic_compute_units)
{
    auto f32 = migraphx::shape::float_type;
    auto x32 = nhwc(f32, {1, 64, 224, 224});
    auto w32 = nhwc(f32, {3, 64, 7, 7});
    EXPECT(not offloads_direct_conv(gfx1201(), x32, w32, {{"padding", {3, 3}}}));
    EXPECT(offloads_direct_conv({"gfx1201", 20}, x32, w32, {{"padding", {3, 3}}}));

    auto f16 = migraphx::shape::half_type;
    auto x16 = nhwc(f16, {1, 16, 480, 480});
    auto w16 = nhwc(f16, {1, 16, 3, 3});
    EXPECT(not offloads_direct_conv(gfx942(), x16, w16, {{"padding", {1, 1}}}));
    EXPECT(offloads_direct_conv({"gfx942", 256}, x16, w16, {{"padding", {1, 1}}}));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
