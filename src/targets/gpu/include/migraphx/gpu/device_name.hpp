/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_GPU_DEVICE_NAME_HPP
#define MIGRAPHX_GUARD_GPU_DEVICE_NAME_HPP

#include <migraphx/gpu/config.hpp>
#include <string>

struct hipDeviceProp_t;

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_GPU_EXPORT std::string get_device_name();

MIGRAPHX_GPU_EXPORT int get_device_id();

MIGRAPHX_GPU_EXPORT bool gfx_has_fp8fnuz_intrinsics();

MIGRAPHX_GPU_EXPORT bool gfx_has_fp8ocp_intrinsics();

MIGRAPHX_GPU_EXPORT bool gfx_has_bf16_intrinsics();

MIGRAPHX_GPU_EXPORT bool gfx_has_fp8fnuz_support();

#if MIGRAPHX_USE_HIPBLASLT
MIGRAPHX_GPU_EXPORT bool gfx_default_rocblas();
#endif

MIGRAPHX_GPU_EXPORT bool hipblaslt_supported();

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_DEVICE_NAME_HPP
