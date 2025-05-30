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
#include <migraphx/env.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/rocblas.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/rank.hpp>
#include <migraphx/stringutils.hpp>
#include <hip/hip_runtime_api.h>

#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_SET_GEMM_PROVIDER)

int get_device_id()
{
    int device;
    auto status = hipGetDevice(&device);
    if(status != hipSuccess)
        MIGRAPHX_THROW("No device");
    return device;
}

std::string get_device_name()
{
    hipDeviceProp_t props{};
    auto status = hipGetDeviceProperties(&props, get_device_id());
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to get device properties");
    return props.gcnArchName;
}

bool gfx_has_fp8fnuz_intrinsics()
{
    const auto device_name = trim(split_string(get_device_name(), ':').front());
    return (starts_with(device_name, "gfx94"));
}

bool gfx_has_fp8ocp_intrinsics()
{
    const auto device_name = trim(split_string(get_device_name(), ':').front());
    bool is_navi_with_fp8ocp = starts_with(device_name, "gfx12") and device_name >= "gfx1200";
    bool is_mi_with_fp8ocp   = starts_with(device_name, "gfx9") and device_name >= "gfx950";
    return (is_navi_with_fp8ocp or is_mi_with_fp8ocp);
}

bool gfx_has_bf16_intrinsics()
{
    const auto device_name = trim(split_string(get_device_name(), ':').front());
    return not(starts_with(device_name, "gfx1030"));
}

#if MIGRAPHX_USE_HIPBLASLT
// Archs that support hipBLASLt but are defaulted to use rocBLAS.
bool gfx_default_rocblas()
{
    const auto device_name = trim(split_string(get_device_name(), ':').front());
    // Default to rocBLAS for gfx90a.
    return ((string_value_of(MIGRAPHX_SET_GEMM_PROVIDER{}) == "hipblaslt")
                ? false
                : (device_name == "gfx90a"));
}
#endif

bool hipblaslt_supported()
{
#if !MIGRAPHX_USE_HIPBLASLT
    return false;
#else
    const auto device_name = trim(split_string(get_device_name(), ':').front());
    // hipblaslt is supported for MI200 and above, and Navi3x and above.
    return (device_name == "gfx90a" or
            (starts_with(device_name, "gfx94") and device_name >= "gfx942") or
            (starts_with(device_name, "gfx95") and device_name >= "gfx950") or
            starts_with(device_name, "gfx110") or starts_with(device_name, "gfx120"));
#endif
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
