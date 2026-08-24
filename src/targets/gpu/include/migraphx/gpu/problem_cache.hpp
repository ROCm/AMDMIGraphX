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
#ifndef MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_HPP
#define MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_HPP

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/gpu/export.h>
#include <migraphx/gpu/cache_device_key.hpp>
#include <migraphx/gpu/problem_cache_backend.hpp>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct context;

struct MIGRAPHX_GPU_EXPORT problem_cache
{
    // Default-constructs with the JSON storage backend.
    problem_cache();

    // Build and store this cache's device key from the owning context.
    void set_device_key(const context& ctx);
    // Directly set the device key (used by tests and multi-cache setup).
    void set_device_key(const cache_device_key& key);
    const cache_device_key& get_device_key() const;

    /// Look up a problem. The writable cache is searched first, then the
    /// read-only caches in priority order; first hit wins.
    bool has(const std::string& name, const value& problem) const;
    void insert(const std::string& name, const value& problem, const value& solution);
    void mark(const std::string& name, const value& problem);
    optional<value> get(const std::string& name, const value& problem) const;
    /// Configure both cache tiers: the read-only caches (searched after the
    /// writable cache, first hit wins, never written) and the read/write caches
    /// (only the first non-empty path is written back to).
    void load(const std::vector<std::string>& read_only_paths,
              const std::vector<std::string>& writable_paths);
    void save() const;

    private:
    // Pluggable storage backend, selected by file type at load() (JSON, or
    // SQLite for a .db/.sqlite path). This is the writable cache: new solutions
    // are inserted and saved here.
    problem_cache_backend backend;

    // Lower-priority read-only layers searched after `backend`; within the list
    // the first hit wins. Populated from the read-only paths passed to load().
    std::vector<problem_cache_backend> read_only_backends{};

    // Device these entries were tuned on; set by the owning context. Empty
    // key = unidentified device, entries land in a single bucket.
    cache_device_key device_key{};

    // File path set by load(path). When non-empty, save() writes here. Empty
    // means no writable cache is configured.
    std::string path_override{};
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_HPP
