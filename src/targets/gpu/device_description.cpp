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
#include <migraphx/gpu/device_description.hpp>
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/gpu/hip.hpp>
#include <migraphx/gpu/hsa_chiplet.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/stringutils.hpp>
#include <hip/hip_runtime_api.h>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// RDNA architectures use wave32
static std::size_t arch_wavefront_size(const std::string& arch_name)
{
    const auto gfx = get_gfx_name(arch_name);
    if(starts_with(gfx, "gfx10") or starts_with(gfx, "gfx11") or starts_with(gfx, "gfx12"))
        return 32;
    return 64;
}

device_description device_description::from_device(std::size_t device)
{
    hipDeviceProp_t props{};
    auto status = hipGetDeviceProperties(&props, device);
    if(status != hipSuccess)
        MIGRAPHX_THROW("Failed to get device properties: " + hip_error(status));

    device_description result;
    result.arch                  = props.gcnArchName;
    result.num_cu                = props.multiProcessorCount;
    result.num_chiplets          = get_hsa_chiplet_count(device);
    result.max_threads_per_cu    = props.maxThreadsPerMultiProcessor;
    result.max_threads_per_block = props.maxThreadsPerBlock;
    result.wavefront_size        = props.warpSize;
    result.last_level_cache_size = get_hsa_last_level_cache_size(device);
    if(result.last_level_cache_size == 0)
        result.last_level_cache_size = std::max(props.l2CacheSize, 0);
    return result;
}

void device_description::normalize()
{
    if(wavefront_size != 0 and wavefront_size != 32 and wavefront_size != 64)
        MIGRAPHX_THROW("Invalid wavefront_size: expected 0 (auto), 32, or 64");

    if(wavefront_size == 0)
        wavefront_size = arch_wavefront_size(arch);
    num_cu                = std::max<std::size_t>(num_cu, 1);
    num_chiplets          = std::max<std::size_t>(num_chiplets, 1);
    max_threads_per_cu    = std::max<std::size_t>(max_threads_per_cu, 1);
    max_threads_per_block = std::max<std::size_t>(max_threads_per_block, 1);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
