/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <unordered_set>
#include <migraphx/ranges.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// for hipblaslt only
static size_t workspace_size = HIPBLASLT_WORKSPACE_SIZE;
static void *d_workspace;

rocblas_handle_ptr create_rocblas_handle_ptr()
{
    rocblas_handle handle;
    rocblas_create_handle(&handle);
    return rocblas_handle_ptr{handle};
}

rocblas_handle_ptr create_rocblas_handle_ptr(hipStream_t s)
{
    rocblas_handle_ptr rb = create_rocblas_handle_ptr();
    rocblas_set_stream(rb.get(), s);
    return rb;
}

hipblaslt_handle_ptr create_hipblaslt_handle_ptr()
{
    hipblasLtHandle_t handle;
    hipblasLtCreate(&handle);
    return hipblaslt_handle_ptr{handle};
}

hipblaslt_handle_ptr create_hipblaslt_handle_ptr(hipStream_t s)
{
    hipblaslt_handle_ptr hblt = create_hipblaslt_handle_ptr();
    hipblasSetStream(hblt.get(), s);
    return hblt;
}

hipblaslt_preference_ptr create_hipblaslt_preference_ptr()
{
    hipblasLtMatmulPreference_t preference;
    hipblasLtMatmulPreferenceCreate(&preference);
    CHECK_HIPBLAS_ERROR(hipblasLtMatmulPreferenceSetAttribute(
        preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));
    return hipblaslt_preference_ptr{preference};
}

hipblaslt_workspace_ptr create_hipblaslt_workspace_ptr()
{
    CHECK_HIP_ERROR(hipMalloc(&d_workspace, workspace_size));
    return hipblaslt_workspace_ptr{d_workspace};
}

bool get_compute_fp32_flag()
{
    const auto device_name = trim(split_string(get_device_name(), ':').front());
    return (starts_with(device_name, "gfx9") and device_name >= "gfx908");
}

bool rocblas_fp8_available()
{
#ifndef MIGRAPHX_USE_ROCBLAS_FP8_API
    return false;
#else
    return gfx_has_fp8_intrinsics();
#endif
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
