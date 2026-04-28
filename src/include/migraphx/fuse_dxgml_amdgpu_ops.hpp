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
#ifndef MIGRAPHX_GUARD_FUSE_DXGML_AMDGPU_OPS_HPP
#define MIGRAPHX_GUARD_FUSE_DXGML_AMDGPU_OPS_HPP

#include <migraphx/config.hpp>
#include <migraphx/module.hpp>
#include <migraphx/export.h>
#include <string>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

/**
 * DxGML-specific pass: fuse dot/add patterns into AMD GPU fused ops.
 *
 * Applied after DxGML MLIR parsing when a kernel registry is configured.
 *
 * Patterns matched:
 *   add(dot(A, B), C)              -> amdgpu::gemm_add(A, B, C)
 *   add(dot(dot(A, B), D), E)      -> amdgpu::gemm_gemm_add(A, B, D, E)
 *
 * Only fires when kernel_registry_file is non-empty.
 */
struct MIGRAPHX_EXPORT fuse_dxgml_amdgpu_ops
{
    /// Path to the JSON kernel registry file.  Embedded as an attribute on
    /// each emitted op so the GPU compiler can look up the code object.
    std::string kernel_registry_file;

    std::string name() const { return "fuse_dxgml_amdgpu_ops"; }
    void apply(module& m) const;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_FUSE_DXGML_AMDGPU_OPS_HPP
