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
#include <migraphx/gpu/json_cache_backend.hpp>
#include <migraphx/json.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/value.hpp>
#include <migraphx/filesystem.hpp>
#include <iostream>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

void json_cache_backend::open(const std::string& path, const cache_device_key& current_device)
{
    filepath_ = path;
    current_device_ = current_device;
    data_.clear();

    if(filepath_.empty())
        return;

    if(not fs::exists(filepath_))
        return;

    auto current_dk_str = to_string(current_device_);

    try
    {
        auto content = read_string(filepath_);
        if(content.empty())
            return;

        // Deserialize JSON file into value map, then project keys to (device_key, name, problem)
        std::unordered_map<value, value> raw;
        from_value(from_json_string(content), raw);
        for(auto& [k, v] : raw)
        {
            auto name    = k.at("name").to<std::string>();
            auto problem = k.at("problem").to<std::string>();
            auto solution = v.is_null() ? std::string{} : v.to<std::string>();

            // Try to reconstruct device_key from stored metadata fields.
            // If the entry has gpu_arch/cu_count/wavefront_size, build a device key.
            // Otherwise (legacy entries), assign to current device.
            std::string dk_str;
            if(k.contains("gpu_arch") && k.contains("cu_count") && k.contains("wavefront_size"))
            {
                cache_device_key stored_dk;
                stored_dk.gpu_arch = k.at("gpu_arch").to<std::string>();
                stored_dk.cu_count = k.at("cu_count").to<int>();
                stored_dk.wavefront_size = k.at("wavefront_size").to<int>();
                dk_str = to_string(stored_dk);
            }
            else
            {
                // Legacy entry without device info → assign to current device
                dk_str = current_dk_str;
            }
            data_[{dk_str, name, problem}] = solution;
        }
    }
    catch(const std::exception& e)
    {
        std::cerr << "[migraphx] WARNING: Failed to parse cache file '"
                  << filepath_ << "': " << e.what()
                  << ". Starting with empty cache." << std::endl;
        data_.clear();
    }
}

void json_cache_backend::close()
{
    // No resources to release for JSON backend
}

bool json_cache_backend::has(const std::string& device_key,
                             const std::string& name,
                             const std::string& problem) const
{
    return data_.count({device_key, name, problem}) > 0;
}

std::optional<std::string> json_cache_backend::get(const std::string& device_key,
                                                   const std::string& name,
                                                   const std::string& problem) const
{
    auto it = data_.find({device_key, name, problem});
    if(it == data_.end())
        return std::nullopt;
    return it->second;
}

void json_cache_backend::insert(const std::string& device_key,
                                const std::string& name,
                                const std::string& problem,
                                const std::string& solution)
{
    data_[{device_key, name, problem}] = solution;
}

void json_cache_backend::mark(const std::string& device_key,
                              const std::string& name,
                              const std::string& problem)
{
    // Only insert if key doesn't already exist (WIP marker)
    data_.emplace(key_type{device_key, name, problem}, std::string{});
}

void json_cache_backend::save()
{
    if(filepath_.empty())
        return;

    // Reconstruct value map with enriched key objects for JSON serialization.
    // Include hardware metadata fields when available — these are stored per-entry
    // for provenance (matches the enriched cache format). Extra fields in the key
    // object do not affect matching: open() extracts only "name", "problem", and device fields.
    std::unordered_map<value, value> output;
    auto current_dk_str = to_string(current_device_);
    for(auto& [k, v] : data_)
    {
        auto& dk_str = std::get<0>(k);
        auto& name = std::get<1>(k);
        auto& problem = std::get<2>(k);

        value key;
        // Parse the entry's own device_key to get its device fields
        auto entry_dk = parse_device_key(dk_str);

        if(!hw_meta_.empty() && dk_str == current_dk_str)
        {
            // Entry belongs to current device — write full hw provenance metadata
            key = {{"name", name},
                   {"problem", problem},
                   {"gpu_arch", hw_meta_.gpu_arch},
                   {"cu_count", hw_meta_.cu_count},
                   {"graphics_clock_mhz", hw_meta_.graphics_clock_mhz},
                   {"memory_clock_mhz", hw_meta_.memory_clock_mhz},
                   {"memory_bus_bits", hw_meta_.memory_bus_bits},
                   {"vram_bytes", hw_meta_.vram_bytes},
                   {"wavefront_size", hw_meta_.wavefront_size},
                   {"regs_per_block", hw_meta_.regs_per_block},
                   {"max_threads_per_cu", hw_meta_.max_threads_per_cu}};
        }
        else if(!entry_dk.empty())
        {
            // Entry belongs to a different device — write its device key fields only
            key = {{"name", name},
                   {"problem", problem},
                   {"gpu_arch", entry_dk.gpu_arch},
                   {"cu_count", entry_dk.cu_count},
                   {"wavefront_size", entry_dk.wavefront_size}};
        }
        else
        {
            // No device info available (legacy entry)
            key = {{"name", name}, {"problem", problem}};
        }
        value sol = v.empty() ? value{} : value(v);
        output[key] = sol;
    }
    write_string(filepath_, to_pretty_json_string(to_value(output)));
}

std::vector<cache_entry> json_cache_backend::all_entries() const
{
    std::vector<cache_entry> entries;
    entries.reserve(data_.size());
    for(auto& [k, v] : data_)
    {
        entries.push_back({std::get<0>(k), std::get<1>(k), std::get<2>(k), v});
    }
    return entries;
}

void json_cache_backend::load_entries(const std::vector<cache_entry>& entries)
{
    auto current_dk_str = to_string(current_device_);
    for(auto& e : entries)
    {
        auto dk = e.device_key.empty() ? current_dk_str : e.device_key;
        data_[{dk, e.name, e.problem}] = e.solution;
    }
}

std::size_t json_cache_backend::size() const { return data_.size(); }

std::string json_cache_backend::backend_name() const { return "json"; }

backend_stats json_cache_backend::stats() const
{
    return {data_.size(), 0, filepath_, "json"};
}

void json_cache_backend::set_hw_metadata(const cache_hw_metadata& meta) { hw_meta_ = meta; }

const cache_hw_metadata& json_cache_backend::get_hw_metadata() const { return hw_meta_; }

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
