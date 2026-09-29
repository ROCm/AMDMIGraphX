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
#include <migraphx/gpu/hsa_chiplet.hpp>
#include <migraphx/errors.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>
#include <type_traits>

#ifndef _WIN32
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

#ifndef _WIN32

namespace {

/// Convert HSA status code to a human-readable string
std::string hsa_error_string(hsa_status_t status)
{
    const char* msg = nullptr;
    if(hsa_status_string(status, &msg) == HSA_STATUS_SUCCESS and msg != nullptr)
        return msg;
    return "Unknown HSA error (code " + std::to_string(static_cast<int>(status)) + ")";
}

/// RAII wrapper for HSA runtime initialization.
/// Calls hsa_init() in constructor and hsa_shut_down() in destructor.
struct hsa_guard
{
    hsa_status_t init_status;
    bool initialized;

    hsa_guard() : init_status(hsa_init()), initialized(init_status == HSA_STATUS_SUCCESS) {}

    ~hsa_guard()
    {
        if(initialized)
            hsa_shut_down();
    }

    hsa_guard(const hsa_guard&)            = delete;
    hsa_guard& operator=(const hsa_guard&) = delete;

    explicit operator bool() const { return initialized; }

    hsa_status_t status() const { return init_status; }
};

struct hsa_gpu_info
{
    std::size_t num_chiplets          = 1;
    std::size_t last_level_cache_size = 0;
};

/// Query the chiplet count and cache sizes for all GPU devices.
std::vector<hsa_gpu_info> query_all_gpu_info()
{
    std::vector<hsa_gpu_info> gpu_infos;

    hsa_guard guard;
    if(not guard)
    {
        MIGRAPHX_THROW("HSA runtime initialization failed: " + hsa_error_string(guard.status()) +
                       ". GPU is not accessible.");
    }

    // HSA agents are enumerated in the same order as HIP device IDs for GPU agents.
    // Reference: ROCm documentation on device enumeration consistency between HIP and HSA.
    auto agent_callback = [&](hsa_agent_t agent) -> hsa_status_t {
        hsa_device_type_t device_type;
        hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
        if(err != HSA_STATUS_SUCCESS)
            return err;

        if(device_type == HSA_DEVICE_TYPE_GPU)
        {
            hsa_gpu_info info;
            uint32_t num_chiplets = 1;
            err                   = hsa_agent_get_info(
                agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_XCC), &num_chiplets);
            // If the query fails (e.g., older ROCm or unsupported GPU), use default of 1.
            // This is expected on older ROCm versions, so no warning needed.
            if(err == HSA_STATUS_SUCCESS)
                info.num_chiplets = num_chiplets;

            // Data cache sizes in bytes for each level, 0 when the level
            // does not exist
            std::array<std::uint32_t, 4> cache_sizes{};
            err = hsa_agent_get_info(agent, HSA_AGENT_INFO_CACHE_SIZE, cache_sizes.data());
            if(err == HSA_STATUS_SUCCESS)
            {
                auto it = std::find_if(
                    cache_sizes.rbegin(), cache_sizes.rend(), [](auto size) { return size != 0; });
                if(it != cache_sizes.rend())
                    info.last_level_cache_size = *it;
            }

            gpu_infos.push_back(info);
        }

        return HSA_STATUS_SUCCESS;
    };

    // Use a non-capturing lambda as the C callback, forwarding to the capturing lambda.
    hsa_status_t status = hsa_iterate_agents(
        [](hsa_agent_t agent, void* user_data) -> hsa_status_t {
            auto* callback = static_cast<std::add_pointer_t<decltype(agent_callback)>>(user_data);
            return (*callback)(agent);
        },
        &agent_callback);
    if(status != HSA_STATUS_SUCCESS and status != HSA_STATUS_INFO_BREAK)
    {
        MIGRAPHX_THROW("HSA agent enumeration failed: " + hsa_error_string(status) +
                       ". Unable to query GPU devices.");
    }

    return gpu_infos;
}

/// Get cached GPU info. Thread-safe, queries HSA only once.
const std::vector<hsa_gpu_info>& get_cached_gpu_info()
{
    static const std::vector<hsa_gpu_info> infos = query_all_gpu_info();
    return infos;
}

} // namespace

std::size_t get_hsa_chiplet_count(std::size_t device_id)
{
    const auto& infos = get_cached_gpu_info();

    if(device_id < infos.size())
        return infos[device_id].num_chiplets;

    // Device not found - HSA enumerated fewer GPUs than expected.
    return 0;
}

std::size_t get_hsa_last_level_cache_size(std::size_t device_id)
{
    const auto& infos = get_cached_gpu_info();

    if(device_id < infos.size())
        return infos[device_id].last_level_cache_size;

    return 0;
}

#else // _WIN32

std::size_t get_hsa_chiplet_count(std::size_t /*device_id*/)
{
    // HSA not available on Windows, assume single chiplet.
    // TODO: For future architectures with multiple chiplets,
    // need a way to query on Windows or hardcode based on gfx number.
    return 1;
}

std::size_t get_hsa_last_level_cache_size(std::size_t /*device_id*/)
{
    // HSA not available on Windows, so the cache size is unknown.
    return 0;
}

#endif // _WIN32

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
