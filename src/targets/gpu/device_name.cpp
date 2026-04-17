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

#include <mutex>
#include <string_view>
#include <unordered_map>
namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_SET_GEMM_PROVIDER)

namespace {

std::string normalize_gfx_name(std::string_view gfx_name)
{
    return trim(split_string(std::string{gfx_name}, ':').front());
}

bool gfx_is_mi3xx_or_newer(std::string_view gfx_name)
{
    const auto device_name = normalize_gfx_name(gfx_name);
    const bool is_gfx94    = starts_with(device_name, "gfx94") and device_name >= "gfx942";
    const bool is_gfx95    = starts_with(device_name, "gfx95") and device_name >= "gfx950";
    return is_gfx94 or is_gfx95;
}

struct cached_device_info
{
    std::string gcn_arch_name{};
    std::string normalized_gfx_name{};
    bool is_navi                 = false;
    bool is_mi3xx_or_newer       = false;
    bool has_fp8fnuz_intrinsics  = false;
    bool has_fp8ocp_intrinsics   = false;
    bool has_bf16_intrinsics     = false;
    bool has_mx_intrinsics       = false;
    bool supports_hipblaslt      = false;
};

cached_device_info read_device_info(int device_id)
{
    hipDeviceProp_t props{};
    auto status = hipGetDeviceProperties(&props, device_id);
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to get device properties");

    cached_device_info info;
    info.gcn_arch_name       = props.gcnArchName;
    info.normalized_gfx_name = normalize_gfx_name(info.gcn_arch_name);
    info.is_navi             = starts_with(info.normalized_gfx_name, "gfx11") or
                   starts_with(info.normalized_gfx_name, "gfx12");
    info.is_mi3xx_or_newer   = gfx_is_mi3xx_or_newer(info.normalized_gfx_name);
    info.has_fp8fnuz_intrinsics = starts_with(info.normalized_gfx_name, "gfx94");
    info.has_fp8ocp_intrinsics  = (starts_with(info.normalized_gfx_name, "gfx12") and
                                  info.normalized_gfx_name >= "gfx1200") or
                                 (starts_with(info.normalized_gfx_name, "gfx9") and
                                  info.normalized_gfx_name >= "gfx950");
    info.has_bf16_intrinsics    = not(starts_with(info.normalized_gfx_name, "gfx1030"));
    info.has_mx_intrinsics      = starts_with(info.normalized_gfx_name, "gfx9") and
                             info.normalized_gfx_name >= "gfx950";
    info.supports_hipblaslt =
        info.normalized_gfx_name == "gfx90a" or info.is_mi3xx_or_newer or
        starts_with(info.normalized_gfx_name, "gfx110") or
        starts_with(info.normalized_gfx_name, "gfx120");
    return info;
}

cached_device_info get_cached_device_info()
{
    static std::mutex cache_mutex;
    static std::unordered_map<int, cached_device_info> cache;

    const auto device_id = get_device_id();
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = cache.find(device_id);
    if(it == cache.end())
        it = cache.emplace(device_id, read_device_info(device_id)).first;
    return it->second;
}

} // namespace

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
    return get_cached_device_info().gcn_arch_name;
}

bool gfx_is_navi(std::string_view gfx_name)
{
    const auto device_name = normalize_gfx_name(gfx_name);
    return starts_with(device_name, "gfx11") or starts_with(device_name, "gfx12");
}

bool gfx_has_fp8fnuz_intrinsics()
{
    return get_cached_device_info().has_fp8fnuz_intrinsics;
}

bool gfx_has_fp8ocp_intrinsics()
{
    return get_cached_device_info().has_fp8ocp_intrinsics;
}

bool gfx_has_bf16_intrinsics()
{
    return get_cached_device_info().has_bf16_intrinsics;
}

bool gfx_has_mx_intrinsics()
{
    return get_cached_device_info().has_mx_intrinsics;
}

bool gfx_prefers_nhwc_layout(std::string_view gfx_name)
{
    return gfx_is_navi(gfx_name) or gfx_is_mi3xx_or_newer(gfx_name);
}

bool gfx_prefers_nhwc_layout()
{
    const auto info = get_cached_device_info();
    return info.is_navi or info.is_mi3xx_or_newer;
}

bool gfx_prefers_mlir_attention(std::string_view gfx_name)
{
    return gfx_is_mi3xx_or_newer(gfx_name);
}

bool gfx_prefers_mlir_attention() { return get_cached_device_info().is_mi3xx_or_newer; }

#if MIGRAPHX_USE_HIPBLASLT
// Archs that support hipBLASLt but are defaulted to use rocBLAS.
bool gfx_default_rocblas()
{
    const auto info = get_cached_device_info();
    // Default to rocBLAS for gfx90a.
    return ((string_value_of(MIGRAPHX_SET_GEMM_PROVIDER{}) == "hipblaslt")
                ? false
                : (info.normalized_gfx_name == "gfx90a"));
}
#endif

bool hipblaslt_supported()
{
#if !MIGRAPHX_USE_HIPBLASLT
    return false;
#else
    return get_cached_device_info().supports_hipblaslt;
#endif
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
