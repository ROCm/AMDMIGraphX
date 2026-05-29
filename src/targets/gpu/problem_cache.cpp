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
 *
 */
#include <migraphx/gpu/problem_cache.hpp>
#include <migraphx/gpu/problem_cache_backend.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/json.hpp>
#include <migraphx/env.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/stringutils.hpp>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <mutex>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_PROBLEM_CACHE)

// Module-scoped backend storage.  Avoids adding any members to the
// problem_cache struct (which would change its ABI / layout and break
// compatibility with the stock migraphx_gpu.dll object files).
// There is at most one active problem_cache per process (owned by
// the gpu::context), so a simple static works.
static problem_cache_backend& active_backend()
{
    static problem_cache_backend backend;
    return backend;
}

// The current device key string (derived from hw metadata on load).
static std::string& active_device_key()
{
    static std::string dk;
    return dk;
}

// Query current GPU hardware properties via HIP.
// Returns empty metadata on failure (no device, HIP not initialized, etc.).
static cache_hw_metadata query_current_gpu_metadata()
{
    cache_hw_metadata meta;
    int device_id = 0;
    if(hipGetDevice(&device_id) != hipSuccess)
        return meta;

    hipDeviceProp_t props{};
    if(hipGetDeviceProperties(&props, device_id) != hipSuccess)
        return meta;

    meta.gpu_arch = props.gcnArchName;
    // Trim target feature flags (e.g. "gfx1100:sramecc+:xnack-" → "gfx1100")
    auto colon = meta.gpu_arch.find(':');
    if(colon != std::string::npos)
        meta.gpu_arch = meta.gpu_arch.substr(0, colon);

    meta.cu_count           = props.multiProcessorCount;
    meta.graphics_clock_mhz = props.clockRate / 1000;       // kHz → MHz
    meta.memory_clock_mhz   = props.memoryClockRate / 1000; // kHz → MHz
    meta.memory_bus_bits    = props.memoryBusWidth;
    meta.vram_bytes         = static_cast<std::int64_t>(props.totalGlobalMem);
    meta.wavefront_size     = props.warpSize;
    meta.regs_per_block     = props.regsPerBlock;
    meta.max_threads_per_cu = props.maxThreadsPerMultiProcessor;

    return meta;
}

void problem_cache::load()
{
    auto& backend = active_backend();
    backend       = make_default_cache_backend();

    auto pc_path = string_value_of(MIGRAPHX_PROBLEM_CACHE{});
    if(pc_path.empty())
        return;

    // Query live GPU hardware metadata to derive device key.
    auto hw_meta        = query_current_gpu_metadata();
    auto dk             = hw_meta.device_key();
    active_device_key() = to_string(dk);

    backend.open(pc_path, dk);

    if(not hw_meta.empty())
        backend.set_hw_metadata(hw_meta);

    // For the JSON backend, populate the legacy in-memory cache for backward
    // compatibility with any code that accesses problem_cache::cache directly.
    if(backend.backend_name() == "json")
    {
        auto entries = backend.all_entries();
        for(auto& e : entries)
        {
            value key  = {{"name", e.name}, {"problem", e.problem}};
            value sol  = e.solution.empty() ? value{} : value(e.solution);
            cache[key] = sol;
        }
    }
}

void problem_cache::load(const std::string& explicit_path, const std::string& explicit_backend)
{
    auto& backend = active_backend();
    backend       = make_cache_backend_with_fallback(explicit_backend);

    // Precedence: explicit path > env var > no cache
    std::string pc_path = explicit_path;
    if(pc_path.empty())
        pc_path = string_value_of(MIGRAPHX_PROBLEM_CACHE{});
    if(pc_path.empty())
        return;

    // Query live GPU hardware metadata to derive device key.
    auto hw_meta        = query_current_gpu_metadata();
    auto dk             = hw_meta.device_key();
    active_device_key() = to_string(dk);

    backend.open(pc_path, dk);

    if(not hw_meta.empty())
        backend.set_hw_metadata(hw_meta);

    // For the JSON backend, populate the legacy in-memory cache for backward
    // compatibility with any code that accesses problem_cache::cache directly.
    if(backend.backend_name() == "json")
    {
        auto entries = backend.all_entries();
        for(auto& e : entries)
        {
            value key  = {{"name", e.name}, {"problem", e.problem}};
            value sol  = e.solution.empty() ? value{} : value(e.solution);
            cache[key] = sol;
        }
    }
}

void problem_cache::save() const
{
    auto& backend = active_backend();
    if(!backend)
        return;

    // For the JSON backend, sync the legacy in-memory cache → backend before
    // persisting, since some code may write to the cache map directly.
    if(backend.backend_name() == "json")
    {
        auto& dk = active_device_key();
        std::vector<cache_entry> entries;
        entries.reserve(cache.size());
        for(auto& [k, v] : cache)
        {
            cache_entry e;
            e.device_key = dk;
            e.name       = k.at("name").to<std::string>();
            e.problem    = k.at("problem").to<std::string>();
            e.solution   = v.is_null() ? std::string{} : v.to<std::string>();
            entries.push_back(std::move(e));
        }
        backend.load_entries(entries);
    }

    backend.save();
}

bool problem_cache::has(const std::string& name, const value& problem) const
{
    const auto& backend = active_backend();
    if(backend)
        return backend.has(active_device_key(), name, problem.to<std::string>());
    value key = {{"name", name}, {"problem", problem}};
    return contains(cache, key);
}

void problem_cache::insert(const std::string& name, const value& problem, const value& solution)
{
    assert(not solution.is_null());
    auto& backend = active_backend();
    if(backend)
        backend.insert(
            active_device_key(), name, problem.to<std::string>(), solution.to<std::string>());

    // Only update legacy cache map for JSON backend (backward compatibility)
    if(!backend || backend.backend_name() == "json")
    {
        value key  = {{"name", name}, {"problem", problem}};
        cache[key] = solution;
    }
}

void problem_cache::mark(const std::string& name, const value& problem)
{
    auto& backend = active_backend();
    if(backend)
        backend.mark(active_device_key(), name, problem.to<std::string>());

    // Only update legacy cache map for JSON backend (backward compatibility)
    if(!backend || backend.backend_name() == "json")
    {
        value key = {{"name", name}, {"problem", problem}};
        cache.insert(std::make_pair(key, value{}));
    }
}

optional<value> problem_cache::get(const std::string& name, const value& problem) const
{
    const auto& backend = active_backend();
    if(backend)
    {
        auto result = backend.get(active_device_key(), name, problem.to<std::string>());
        if(!result)
            return nullopt;
        if(result->empty())
            return value{};
        return value(*result);
    }
    value key = {{"name", name}, {"problem", problem}};
    auto it   = cache.find(key);
    if(it == cache.end())
        return nullopt;
    return it->second;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
