/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef MIGRAPHX_GUARD_MIGRAPHLIB_HIPBLASLT_HPP
#define MIGRAPHX_GUARD_MIGRAPHLIB_HIPBLASLT_HPP
#include <migraphx/argument.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/gpu/config.hpp>
#include <migraphx/errors.hpp>
#if MIGRAPHX_USE_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>

// TODO: Remove hipblasStatusToString() function when hipblaslt
// provides an API for doing this in hipBLASLt.

// Convert hipblas_status to string
inline const char* hipblasStatusToString(hipblasStatus_t status)
{
#define CASE(x) \
    case x:     \
        return #x
    switch(status)
    {
        CASE(HIPBLAS_STATUS_SUCCESS);
        CASE(HIPBLAS_STATUS_NOT_INITIALIZED);
        CASE(HIPBLAS_STATUS_ALLOC_FAILED);
        CASE(HIPBLAS_STATUS_INVALID_VALUE);
        CASE(HIPBLAS_STATUS_MAPPING_ERROR);
        CASE(HIPBLAS_STATUS_EXECUTION_FAILED);
        CASE(HIPBLAS_STATUS_INTERNAL_ERROR);
        CASE(HIPBLAS_STATUS_NOT_SUPPORTED);
        CASE(HIPBLAS_STATUS_ARCH_MISMATCH);
        CASE(HIPBLAS_STATUS_HANDLE_IS_NULLPTR);
        CASE(HIPBLAS_STATUS_INVALID_ENUM);
        CASE(HIPBLAS_STATUS_UNKNOWN);
    }
#undef CASE
    // We don't use default: so that the compiler warns us if any valid enums are missing
    // from our switch. If the value is not a valid hipblas_status, we return this string.
    return "<undefined hipblasStatus_t value>";
}

template <class F, class... Ts>
inline auto hipblaslt_invoke(F f, Ts... xs)
{
    // Call the function `f` with `xs...` and capture the status
    auto status = f(xs...);

    if(status != HIPBLAS_STATUS_SUCCESS)
    {
        std::string error_message =
            "hipBLAS error: '" + std::string(hipblasStatusToString(status)) + "'(" +
            std::to_string(status) + ") at " + __FILE__ + ":" + std::to_string(__LINE__);
        MIGRAPHX_THROW(EXIT_FAILURE, error_message);
    }
    return status;
}

template <class F, class Pack, class... Ts>
auto hipblaslt_invoke(F f, Pack p, Ts... xs)
{
    return p([=](auto... ws) {
        auto status = f(ws..., xs...);
        if(status != HIPBLAS_STATUS_SUCCESS)
        {
            MIGRAPHX_THROW("hipblaslt_invoke: hipBlasLt call failed with status " +
                           std::to_string(status));
        }
        return status;
    });
}

#endif // MIGRAPHX_USE_HIPBLASLT

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

#if MIGRAPHX_USE_HIPBLASLT
using hipblaslt_handle_ptr     = MIGRAPHX_MANAGE_PTR(hipblasLtHandle_t, hipblasLtDestroy);
using hipblaslt_preference_ptr = MIGRAPHX_MANAGE_PTR(hipblasLtMatmulPreference_t,
                                                     hipblasLtMatmulPreferenceDestroy);

hipblaslt_handle_ptr create_hipblaslt_handle_ptr();
hipblaslt_preference_ptr create_hipblaslt_preference_ptr();
bool hipblaslt_supported();
const size_t hipblaslt_workspace_size = 2 * 128 * 1024 * 1024;
#endif // MIGRAPHX_USE_HIPBLASLT

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_MIGRAPHLIB_HIPBLASLT_HPP
