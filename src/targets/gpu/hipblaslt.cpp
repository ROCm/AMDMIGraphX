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

#include <migraphx/ranges.hpp>
#include <migraphx/gpu/hipblaslt.hpp>
#include <migraphx/gpu/context.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {
#if MIGRAPHX_USE_HIPBLASLT
// for hipblaslt only
static const size_t workspace_size = hipblaslt_workspace_size;

hipblaslt_handle_ptr create_hipblaslt_handle_ptr()
{
    hipblasLtHandle_t handle;
    hipblasLtCreate(&handle);
    return hipblaslt_handle_ptr{handle};
}

hipblaslt_preference_ptr create_hipblaslt_preference_ptr()
{
    hipblasLtMatmulPreference_t preference;
    hipblasLtMatmulPreferenceCreate(&preference);
    hipblaslt_invoke([&]() {
        return hipblasLtMatmulPreferenceSetAttribute(preference,
                                                     HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                     &workspace_size,
                                                     sizeof(workspace_size));
    });
    return hipblaslt_preference_ptr{preference};
}

#endif // MIGRAPHX_USE_HIPBLASLT

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
