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
#include <migraphx/gpu/device_name.hpp>
#include <test.hpp>

TEST_CASE(gfx_is_navi_detects_client_arches)
{
    EXPECT(migraphx::gpu::gfx_is_navi("gfx1100"));
    EXPECT(migraphx::gpu::gfx_is_navi("gfx1100:sramecc+:xnack-"));
    EXPECT(migraphx::gpu::gfx_is_navi("gfx1201"));
    EXPECT(not migraphx::gpu::gfx_is_navi("gfx942"));
    EXPECT(not migraphx::gpu::gfx_is_navi("gfx1030"));
}

TEST_CASE(gfx_prefers_nhwc_layout_modern_arches)
{
    EXPECT(migraphx::gpu::gfx_prefers_nhwc_layout("gfx1100"));
    EXPECT(migraphx::gpu::gfx_prefers_nhwc_layout("gfx1200"));
    EXPECT(migraphx::gpu::gfx_prefers_nhwc_layout("gfx942:sramecc+:xnack-"));
    EXPECT(migraphx::gpu::gfx_prefers_nhwc_layout("gfx942"));
    EXPECT(migraphx::gpu::gfx_prefers_nhwc_layout("gfx950"));
    EXPECT(not migraphx::gpu::gfx_prefers_nhwc_layout("gfx1030"));
    EXPECT(not migraphx::gpu::gfx_prefers_nhwc_layout("gfx940"));
}

TEST_CASE(gfx_prefers_mlir_attention_modern_mi_arches)
{
    EXPECT(migraphx::gpu::gfx_prefers_mlir_attention("gfx942"));
    EXPECT(migraphx::gpu::gfx_prefers_mlir_attention("gfx950"));
    EXPECT(not migraphx::gpu::gfx_prefers_mlir_attention("gfx940"));
    EXPECT(not migraphx::gpu::gfx_prefers_mlir_attention("gfx1100"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
