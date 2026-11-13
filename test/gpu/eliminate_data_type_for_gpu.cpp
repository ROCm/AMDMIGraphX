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
#include <migraphx/gpu/eliminate_data_type_for_gpu.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/module.hpp>
#include <migraphx/pass_manager.hpp>

#include <test.hpp>

static void run_pass(migraphx::module& m, const migraphx::gpu::eliminate_data_type_for_gpu& p)
{
    migraphx::run_passes(m, {p, migraphx::dead_code_elimination{}});
}

TEST_CASE(keep_fp8_quant_dot_for_mlir)
{
    migraphx::gpu::device_description desc;
    desc.arch = "gfx940";
    migraphx::gpu::context ctx{desc};

    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::fp8e4m3fnuz_type, {2, 2}});
        auto y   = m1.add_parameter("y", {migraphx::shape::fp8e4m3fnuz_type, {2, 2}});
        auto dot = m1.add_instruction(migraphx::make_op("quant_dot"), x, y);
        m1.add_return({dot});
    }
    migraphx::module m2 = m1;
    run_pass(m1, {.ctx = &ctx, .mlir_all_dots = true});
    EXPECT(m1 == m2);
}

TEST_CASE(convert_fp8_quant_dot_without_mlir)
{
    migraphx::gpu::device_description desc;
    desc.arch = "gfx940";
    migraphx::gpu::context ctx{desc};

    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::fp8e4m3fnuz_type, {2, 2}});
        auto y   = m1.add_parameter("y", {migraphx::shape::fp8e4m3fnuz_type, {2, 2}});
        auto dot = m1.add_instruction(migraphx::make_op("quant_dot"), x, y);
        m1.add_return({dot});
    }
    run_pass(m1, {.ctx = &ctx, .mlir_all_dots = false});

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::fp8e4m3fnuz_type, {2, 2}});
        auto y  = m2.add_parameter("y", {migraphx::shape::fp8e4m3fnuz_type, {2, 2}});
        auto fx = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto fy = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), y);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), fx, fy);
        auto out = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), dot);
        m2.add_return({out});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(convert_fp8_dot_for_mlir)
{
    migraphx::gpu::device_description desc;
    desc.arch = "gfx940";
    migraphx::gpu::context ctx{desc};

    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::fp8e4m3fnuz_type, {2, 2}});
        auto y   = m1.add_parameter("y", {migraphx::shape::fp8e4m3fnuz_type, {2, 2}});
        auto dot = m1.add_instruction(migraphx::make_op("dot"), x, y);
        m1.add_return({dot});
    }
    run_pass(m1, {.ctx = &ctx, .mlir_all_dots = true});

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::fp8e4m3fnuz_type, {2, 2}});
        auto y  = m2.add_parameter("y", {migraphx::shape::fp8e4m3fnuz_type, {2, 2}});
        auto fx = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto fy = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), y);
        auto dot = m2.add_instruction(migraphx::make_op("dot"), fx, fy);
        auto out = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::fp8e4m3fnuz_type}}),
            dot);
        m2.add_return({out});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(keep_ocp_fp8_quant_dot_for_mlir_gfx1170)
{
    migraphx::gpu::device_description desc;
    desc.arch = "gfx1170";
    migraphx::gpu::context ctx{desc};

    migraphx::module m1;
    {
        auto x   = m1.add_parameter("x", {migraphx::shape::fp8e4m3fn_type, {2, 2}});
        auto y   = m1.add_parameter("y", {migraphx::shape::fp8e4m3fn_type, {2, 2}});
        auto dot = m1.add_instruction(migraphx::make_op("quant_dot"), x, y);
        m1.add_return({dot});
    }
    migraphx::module m2 = m1;
    run_pass(m1, {.ctx = &ctx, .mlir_all_dots = true});
    EXPECT(m1 == m2);
}

TEST_CASE(keep_ocp_fp8_convolution_for_mlir_gfx1170)
{
    migraphx::gpu::device_description desc;
    desc.arch = "gfx1170";
    migraphx::gpu::context ctx{desc};

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::fp8e4m3fn_type, {1, 1, 2, 2}});
        auto w    = m1.add_parameter("w", {migraphx::shape::fp8e4m3fn_type, {1, 1, 1, 1}});
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    migraphx::module m2 = m1;
    run_pass(m1, {.ctx = &ctx, .mlir_convolutions = true});
    EXPECT(m1 == m2);
}

#if MIGRAPHX_USE_MIOPEN
TEST_CASE(convert_ocp_fp8_convolution_without_mlir_gfx1170)
{
    migraphx::gpu::device_description desc;
    desc.arch = "gfx1170";
    migraphx::gpu::context ctx{desc};

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::fp8e4m3fn_type, {1, 1, 2, 2}});
        auto w    = m1.add_parameter("w", {migraphx::shape::fp8e4m3fn_type, {1, 1, 1, 1}});
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    run_pass(m1, {.ctx = &ctx, .mlir_convolutions = false});

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::fp8e4m3fn_type, {1, 1, 2, 2}});
        auto w  = m2.add_parameter("w", {migraphx::shape::fp8e4m3fn_type, {1, 1, 1, 1}});
        auto fx = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto fw = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), w);
        auto conv = m2.add_instruction(migraphx::make_op("convolution"), fx, fw);
        auto out  = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::fp8e4m3fn_type}}), conv);
        m2.add_return({out});
    }
    EXPECT(m1 == m2);
}

TEST_CASE(convert_fnuz_fp8_convolution_without_mlir_gfx940)
{
    migraphx::gpu::device_description desc;
    desc.arch = "gfx940";
    migraphx::gpu::context ctx{desc};

    migraphx::module m1;
    {
        auto x    = m1.add_parameter("x", {migraphx::shape::fp8e4m3fnuz_type, {1, 1, 2, 2}});
        auto w    = m1.add_parameter("w", {migraphx::shape::fp8e4m3fnuz_type, {1, 1, 1, 1}});
        auto conv = m1.add_instruction(migraphx::make_op("convolution"), x, w);
        m1.add_return({conv});
    }
    run_pass(m1, {.ctx = &ctx, .mlir_convolutions = false});

    migraphx::module m2;
    {
        auto x  = m2.add_parameter("x", {migraphx::shape::fp8e4m3fnuz_type, {1, 1, 2, 2}});
        auto w  = m2.add_parameter("w", {migraphx::shape::fp8e4m3fnuz_type, {1, 1, 1, 1}});
        auto fx = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), x);
        auto fw = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::float_type}}), w);
        auto conv = m2.add_instruction(migraphx::make_op("convolution"), fx, fw);
        auto out  = m2.add_instruction(
            migraphx::make_op("convert", {{"target_type", migraphx::shape::fp8e4m3fnuz_type}}),
            conv);
        m2.add_return({out});
    }
    EXPECT(m1 == m2);
}
#endif

int main(int argc, const char* argv[]) { test::run(argc, argv); }
