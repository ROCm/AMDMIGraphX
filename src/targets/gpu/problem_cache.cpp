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
#include <migraphx/gpu/device_name.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/json.hpp>
#include <migraphx/env.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/stringutils.hpp>
#include <hip/hip_runtime_api.h>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_PROBLEM_CACHE)

static value create_key(const std::string& name, const value& problem)
{
    return {{"name", name}, {"problem", problem}};
}

void problem_cache::load()
{
    auto pc_path = string_value_of(MIGRAPHX_PROBLEM_CACHE{});
    if(pc_path.empty())
        return;
    if(not fs::exists(pc_path))
    {
        log::info() << "Problem cache not found. Creating new file.";
        save();
        return;
    }
    // Deserialize into a temporary map, then project keys to {name, problem}
    // so that extra metadata fields in the JSON don't break key matching.
    std::unordered_map<value, value> raw;
    from_value(from_json_string(read_string(pc_path)), raw);
    for(auto& [k, v] : raw)
    {
        auto projected = create_key(k.at("name").to<std::string>(), k.at("problem"));
        cache[projected] = v;
    }
}
void problem_cache::save() const
{
    auto pc_path = string_value_of(MIGRAPHX_PROBLEM_CACHE{});
    if(pc_path.empty())
        return;
    // Enrich keys with hardware provenance metadata on write.
    // This runs once at session end — negligible cost.
    hipDeviceProp_t props{};
    auto status = hipGetDeviceProperties(&props, get_device_id());

    std::unordered_map<value, value> enriched;
    for(auto& [k, v] : cache)
    {
        value rich_key = k;
        if(status == hipSuccess)
        {
            rich_key["gpu_arch"]            = trim(split_string(std::string(props.gcnArchName), ':').front());
            rich_key["cu_count"]            = static_cast<std::int64_t>(props.multiProcessorCount);
            rich_key["graphics_clock_mhz"]  = static_cast<std::int64_t>(props.clockRate / 1000);
            rich_key["memory_clock_mhz"]    = static_cast<std::int64_t>(props.memoryClockRate / 1000);
            rich_key["memory_bus_bits"]      = static_cast<std::int64_t>(props.memoryBusWidth);
            rich_key["vram_bytes"]           = static_cast<std::int64_t>(props.totalGlobalMem);
            rich_key["wavefront_size"]       = static_cast<std::int64_t>(props.warpSize);
            rich_key["regs_per_block"]       = static_cast<std::int64_t>(props.regsPerBlock);
            rich_key["max_threads_per_cu"]   = static_cast<std::int64_t>(props.maxThreadsPerMultiProcessor);
        }
        enriched[rich_key] = v;
    }
    write_string(pc_path, to_pretty_json_string(to_value(enriched)));
}

bool problem_cache::has(const std::string& name, const value& problem) const
{
    return contains(cache, create_key(name, problem));
}

void problem_cache::insert(const std::string& name, const value& problem, const value& solution)
{
    assert(not solution.is_null());
    cache[create_key(name, problem)] = solution;
}

void problem_cache::mark(const std::string& name, const value& problem)
{
    cache.insert(std::make_pair(create_key(name, problem), value{}));
}

optional<value> problem_cache::get(const std::string& name, const value& problem) const
{
    auto it = cache.find(create_key(name, problem));
    if(it == cache.end())
        return nullopt;
    return it->second;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
