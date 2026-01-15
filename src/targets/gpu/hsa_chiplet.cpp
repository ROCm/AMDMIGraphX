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
#include <vector>
#include <mutex>

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

/// Query chiplet counts for all GPU devices and cache the results.
/// This is called once and the results are stored in a static vector.
std::vector<std::size_t> query_all_chiplet_counts()
{
    std::vector<std::size_t> chiplet_counts;

    hsa_guard guard;
    if(not guard)
    {
        MIGRAPHX_THROW("HSA runtime initialization failed: " + hsa_error_string(guard.status()) +
                       ". GPU is not accessible.");
    }

    // Structure to collect chiplet counts for all GPUs
    struct agent_data
    {
        std::vector<std::size_t>* counts;
    };

    agent_data data{&chiplet_counts};

    // Callback function for hsa_iterate_agents.
    // HSA agents are enumerated in the same order as HIP device IDs for GPU agents.
    // Reference: ROCm documentation on device enumeration consistency between HIP and HSA.
    auto agent_callback = [](hsa_agent_t agent, void* user_data) -> hsa_status_t {
        auto* agent_data_ptr = static_cast<agent_data*>(user_data);

        hsa_device_type_t device_type;
        hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
        if(err != HSA_STATUS_SUCCESS)
            return err;

        if(device_type == HSA_DEVICE_TYPE_GPU)
        {
            uint32_t num_chiplets = 1;
            err                   = hsa_agent_get_info(
                agent, static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_XCC), &num_chiplets);
            // If the query fails (e.g., older ROCm or unsupported GPU), use default of 1.
            // This is expected on older ROCm versions, so no warning needed.
            if(err != HSA_STATUS_SUCCESS)
                num_chiplets = 1;

            agent_data_ptr->counts->push_back(static_cast<std::size_t>(num_chiplets));
        }

        return HSA_STATUS_SUCCESS;
    };

    hsa_status_t status = hsa_iterate_agents(agent_callback, &data);
    if(status != HSA_STATUS_SUCCESS and status != HSA_STATUS_INFO_BREAK)
    {
        MIGRAPHX_THROW("HSA agent enumeration failed: " + hsa_error_string(status) +
                       ". Unable to query GPU devices.");
    }

    return chiplet_counts;
}

/// Get cached chiplet counts. Thread-safe, queries HSA only once.
const std::vector<std::size_t>& get_cached_chiplet_counts()
{
    static std::once_flag flag;
    static std::vector<std::size_t> counts;

    std::call_once(flag, []() { counts = query_all_chiplet_counts(); });

    return counts;
}

} // namespace

std::size_t get_hsa_chiplet_count(std::size_t device_id)
{
    const auto& counts = get_cached_chiplet_counts();

    if(device_id < counts.size())
        return counts[device_id];

    // Device not found - HSA enumerated fewer GPUs than expected.
    // This shouldn't happen in normal operation, but return default 1.
    return 1;
}

#else // _WIN32

std::size_t get_hsa_chiplet_count(std::size_t /*device_id*/)
{
    // HSA not available on Windows, assume single chiplet.
    // TODO: For future architectures with multiple chiplets,
    // need a way to query on Windows or hardcode based on gfx number.
    return 1;
}

#endif // _WIN32

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
