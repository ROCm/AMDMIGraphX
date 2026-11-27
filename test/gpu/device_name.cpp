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

#include <migraphx/gpu/context.hpp>
#include <migraphx/gpu/device_name.hpp>

#include <test.hpp>

TEST_CASE(fp8ocp_intrinsics_gfx117x)
{
    migraphx::gpu::device_description desc;
    desc.arch = "gfx1170";
    EXPECT(migraphx::gpu::get_gfx_generation("gfx942") == 9);
    EXPECT(migraphx::gpu::get_gfx_generation("gfx950") == 9);
    EXPECT(migraphx::gpu::get_gfx_generation("gfx1030") == 10);
    EXPECT(migraphx::gpu::get_gfx_generation("gfx1151") == 11);
    EXPECT(migraphx::gpu::get_gfx_generation("gfx1200") == 12);
    EXPECT(migraphx::gpu::get_gfx_generation("gfx1250") == 12);
}

TEST_CASE(gfx_generation_with_features)
{
    EXPECT(migraphx::gpu::gfx_has_fp8ocp_intrinsics(migraphx::gpu::context{desc}));
    desc.arch = "gfx1171";
    EXPECT(migraphx::gpu::gfx_has_fp8ocp_intrinsics(migraphx::gpu::context{desc}));
    desc.arch = "gfx1172";

TEST_CASE(gfx_generation_suffixed_processor)
{
    // gfx1250-strict is a distinct target for codegen but the same ISA generation as gfx1250;
    // clang predefines __GFX12__ for both.
    EXPECT(migraphx::gpu::get_gfx_generation("gfx1250-strict") == 12);
    EXPECT(migraphx::gpu::gfx_has_fp8ocp_intrinsics(migraphx::gpu::context{desc}));
}

TEST_CASE(fp8ocp_intrinsics_other_gfx11)
{
    migraphx::gpu::device_description desc;
    desc.arch = "gfx1100";
    EXPECT(not migraphx::gpu::gfx_has_fp8ocp_intrinsics(migraphx::gpu::context{desc}));
    desc.arch = "gfx1151";

TEST_CASE(gfx_name_strips_features_only)
{
    EXPECT(migraphx::gpu::get_gfx_name("gfx1250") == "gfx1250");
    EXPECT(migraphx::gpu::get_gfx_name("gfx942:sramecc+:xnack-") == "gfx942");
    // The '-strict' suffix must survive: it is what keeps code objects and cache keys for A0
    // distinct from gfx1250.
    EXPECT(migraphx::gpu::get_gfx_name("gfx1250-strict") == "gfx1250-strict");
    EXPECT(not migraphx::gpu::gfx_has_fp8ocp_intrinsics(migraphx::gpu::context{desc}));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
