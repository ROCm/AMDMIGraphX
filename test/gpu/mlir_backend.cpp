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

#include <migraphx/errors.hpp>
#include <migraphx/gpu/mlir.hpp>
#include <test.hpp>

TEST_CASE(mlir_backend_auto_uses_triton_on_gfx117x)
{
    EXPECT(migraphx::gpu::select_mlir_backend("auto", "gfx1170") == "triton");
    EXPECT(migraphx::gpu::select_mlir_backend("auto", "gfx1172") == "triton");
}

TEST_CASE(mlir_backend_auto_uses_legacy_on_other_archs)
{
    EXPECT(migraphx::gpu::select_mlir_backend("auto", "gfx1100") == "legacy");
    EXPECT(migraphx::gpu::select_mlir_backend("auto", "gfx1173") == "legacy");
}

TEST_CASE(mlir_backend_empty_request_is_auto)
{
    EXPECT(migraphx::gpu::select_mlir_backend("", "gfx1171") == "triton");
    EXPECT(migraphx::gpu::select_mlir_backend("", "gfx1201") == "legacy");
}

TEST_CASE(mlir_backend_explicit_request_overrides_auto)
{
    EXPECT(migraphx::gpu::select_mlir_backend("legacy", "gfx1170") == "legacy");
    EXPECT(migraphx::gpu::select_mlir_backend("triton", "gfx1100") == "triton");
}

TEST_CASE(mlir_backend_invalid_request_throws)
{
    EXPECT(test::throws<migraphx::exception>(
        [] { migraphx::gpu::select_mlir_backend("rocmlir", "gfx1100"); },
        "Invalid MIGRAPHX_MLIR_BACKEND value 'rocmlir'"));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
