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

static std::string get_gfx_name(const std::string& device_name)
{
    return trim(split_string(device_name, ':').front());
}

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

bool gfx_has_fp8fnuz_intrinsics(const std::string& device_name)
{
    const auto gfx_name = get_gfx_name(device_name);
    return (starts_with(gfx_name, "gfx94"));
}

bool gfx_has_fp8fnuz_intrinsics() { return gfx_has_fp8fnuz_intrinsics(get_device_name()); }

bool gfx_has_fp8ocp_intrinsics(const std::string& device_name)
{
    const auto gfx_name      = get_gfx_name(device_name);
    bool is_navi_with_fp8ocp = starts_with(gfx_name, "gfx12") and gfx_name >= "gfx1200";
    bool is_mi_with_fp8ocp   = starts_with(gfx_name, "gfx9") and gfx_name >= "gfx950";
    return (is_navi_with_fp8ocp or is_mi_with_fp8ocp);
}

bool gfx_has_fp8ocp_intrinsics() { return gfx_has_fp8ocp_intrinsics(get_device_name()); }

bool gfx_has_bf16_intrinsics(const std::string& device_name)
{
    const auto gfx_name = get_gfx_name(device_name);
    return not(starts_with(gfx_name, "gfx1030"));
}

bool gfx_has_bf16_intrinsics() { return gfx_has_bf16_intrinsics(get_device_name()); }

bool gfx_has_mx_intrinsics(const std::string& device_name)
{
    const auto gfx_name = get_gfx_name(device_name);
    return starts_with(gfx_name, "gfx9") and gfx_name >= "gfx950";
}

bool gfx_has_mx_intrinsics() { return gfx_has_mx_intrinsics(get_device_name()); }

#if MIGRAPHX_USE_HIPBLASLT
bool gfx_default_rocblas(const std::string& device_name)
{
    const auto gfx_name = get_gfx_name(device_name);
    return ((string_value_of(MIGRAPHX_SET_GEMM_PROVIDER{}) == "hipblaslt")
                ? false
                : (gfx_name == "gfx90a"));
}

bool gfx_default_rocblas() { return gfx_default_rocblas(get_device_name()); }
#endif

bool hipblaslt_supported(const std::string& device_name)
{
#if !MIGRAPHX_USE_HIPBLASLT
    (void)device_name;
    return false;
#else
    const auto gfx_name = get_gfx_name(device_name);
    return (gfx_name == "gfx90a" or (starts_with(gfx_name, "gfx94") and gfx_name >= "gfx942") or
            (starts_with(gfx_name, "gfx95") and gfx_name >= "gfx950") or
            starts_with(gfx_name, "gfx110") or starts_with(gfx_name, "gfx120"));
#endif
}

bool hipblaslt_supported() { return hipblaslt_supported(get_device_name()); }

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
