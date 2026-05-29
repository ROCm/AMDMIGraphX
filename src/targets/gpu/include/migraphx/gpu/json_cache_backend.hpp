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
#ifndef MIGRAPHX_GUARD_GPU_JSON_CACHE_BACKEND_HPP
#define MIGRAPHX_GUARD_GPU_JSON_CACHE_BACKEND_HPP

#include <migraphx/config.hpp>
#include <migraphx/gpu/problem_cache_backend.hpp>
#include <unordered_map>
#include <string>
#include <tuple>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

/// JSON file-based cache backend (satisfies problem_cache_backend concept).
///
/// This is the default backend that preserves the original problem_cache behavior:
///   - open(): reads a JSON file into an in-memory map, partitioned by device_key
///   - save(): writes the in-memory map back to the JSON file with hw metadata
///   - has/get/insert/mark: operate on the in-memory map using (device_key, name, problem)
///
/// Legacy JSON files (no device_key) are loaded under the current device key.
class json_cache_backend
{
    public:
    void open(const std::string& path, const cache_device_key& current_device);
    void close();

    bool
    has(const std::string& device_key, const std::string& name, const std::string& problem) const;
    std::optional<std::string>
    get(const std::string& device_key, const std::string& name, const std::string& problem) const;

    void insert(const std::string& device_key,
                const std::string& name,
                const std::string& problem,
                const std::string& solution);
    void mark(const std::string& device_key, const std::string& name, const std::string& problem);

    void save();

    std::vector<cache_entry> all_entries() const;
    void load_entries(const std::vector<cache_entry>& entries);

    std::size_t size() const;
    std::string backend_name() const;
    backend_stats stats() const;

    void set_hw_metadata(const cache_hw_metadata& meta);
    const cache_hw_metadata& get_hw_metadata() const;

    private:
    // Key: (device_key_string, name, problem)
    using key_type = std::tuple<std::string, std::string, std::string>;

    struct key_hash
    {
        std::size_t operator()(const key_type& k) const
        {
            auto h1 = std::hash<std::string>{}(std::get<0>(k));
            auto h2 = std::hash<std::string>{}(std::get<1>(k));
            auto h3 = std::hash<std::string>{}(std::get<2>(k));
            return h1 ^ (h2 << 1U) ^ (h3 << 2U);
        }
    };

    std::string filepath_;
    cache_device_key current_device_;
    cache_hw_metadata hw_meta_;
    std::unordered_map<key_type, std::string, key_hash> data_;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_JSON_CACHE_BACKEND_HPP
