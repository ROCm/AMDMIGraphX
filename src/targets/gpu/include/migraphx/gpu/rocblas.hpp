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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_ROCBLAS_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_ROCBLAS_HPP
#include <migraphx/manage_ptr.hpp>
#include <migraphx/gpu/config.hpp>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>

#define HIPBLASLT_WORKSPACE_SIZE (2*128*1024*1024)

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif

#ifndef CHECK_HIPBLAS_ERROR
#define CHECK_HIPBLAS_ERROR(error)                    \
    if(error != HIPBLAS_STATUS_SUCCESS)               \
    {                                                 \
        fprintf(stderr,                               \
                "hipBLAS error: '%s'(%d) at %s:%d\n", \
                hipblasStatusToString(error),         \
                error,                                \
                __FILE__,                             \
                __LINE__);                            \
        exit(EXIT_FAILURE);                           \
    }
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

using rocblas_handle_ptr = MIGRAPHX_MANAGE_PTR(rocblas_handle, rocblas_destroy_handle);
rocblas_handle_ptr create_rocblas_handle_ptr();
rocblas_handle_ptr create_rocblas_handle_ptr(hipStream_t s);

using hipblaslt_handle_ptr = MIGRAPHX_MANAGE_PTR(hipblasLtHandle_t, hipblasLtDestroy);
hipblaslt_handle_ptr create_hipblaslt_handle_ptr();
hipblaslt_handle_ptr create_hipblaslt_handle_ptr(hipStream_t s);

using hipblaslt_preference_ptr = MIGRAPHX_MANAGE_PTR(hipblasLtMatmulPreference_t, hipblasLtMatmulPreferenceDestroy);
hipblaslt_preference_ptr create_hipblaslt_preference_ptr();

using hipblaslt_workspace_ptr = MIGRAPHX_MANAGE_PTR(void *, hipFree);
hipblaslt_workspace_ptr create_hipblaslt_workspace_ptr();

struct context;

MIGRAPHX_GPU_EXPORT bool get_compute_fp32_flag();

MIGRAPHX_GPU_EXPORT bool rocblas_fp8_available();

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif
