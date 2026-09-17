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
#include <migraphx/gpu/json_problem_cache.hpp>
#include <migraphx/gpu/sqlite_problem_cache.hpp>
#include <migraphx/gpu/context.hpp>
#include <migraphx/stringutils.hpp>
#include <algorithm>
#include <cassert>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Seed the writable set with an in-memory JSON cache so lookups and inserts work
// before (or without) any load(); load() replaces it with the configured
// writable files. save() stays a no-op until a writable file path is set.
problem_cache::problem_cache() { writable_backends.emplace_back(json_problem_cache{}); }

// Select the storage backend by file type: a ".db"/".sqlite" path uses the
// SQLite backend, anything else (including no extension) uses JSON. This lets a
// priority list mix cache formats.
static problem_cache_backend make_problem_cache_backend(const std::string& path)
{
    if(ends_with(path, ".db") or ends_with(path, ".sqlite"))
        return problem_cache_backend{sqlite_problem_cache{}};
    return problem_cache_backend{json_problem_cache{}};
}

static value create_key(const std::string& name, const value& problem)
{
    return {{"name", name}, {"problem", problem.normalize()}};
}

void problem_cache::set_device_key(const context& ctx)
{
    const auto& dev        = ctx.get_current_device();
    const std::string arch = dev.get_device_name();
    if(arch.empty())
    {
        device_key = {};
        return;
    }
    device_key.device_name    = arch;
    device_key.gfx_name       = dev.get_gfx_name();
    device_key.cu_count       = dev.get_cu_count();
    device_key.wavefront_size = dev.get_wavefront_size();
}

void problem_cache::set_device_key(const cache_device_key& key) { device_key = key; }

const cache_device_key& problem_cache::get_device_key() const { return device_key; }

void problem_cache::save() const
{
    // A no-op when no writable file paths are configured (in-memory or read-only
    // cache); otherwise each writable cache saves to its own file.
    for(std::size_t i = 0; i < save_paths.size(); ++i)
        writable_backends[i].save(save_paths[i]);
}

void problem_cache::load(const std::vector<std::string>& read_only_paths,
                         const std::vector<std::string>& writable_paths)
{
    read_only_backends.clear();
    writable_backends.clear();
    save_paths.clear();

    // Read/write caches (the developer caches, the common tuning case): every
    // file is loaded and saved back to. Pick each backend by file type.
    for(const auto& path : writable_paths)
    {
        if(path.empty())
            continue;
        save_paths.push_back(path);
        writable_backends.push_back(make_problem_cache_backend(path));
        writable_backends.back().load(path);
    }
    // Keep an in-memory JSON cache when no writable file is configured so new
    // solutions still have somewhere to go; save() then stays a no-op.
    if(writable_backends.empty())
        writable_backends.emplace_back(json_problem_cache{});

    // Read-only caches (e.g. system-level caches shipped by gpuep or an ISV):
    // searched after the writable caches and never written back.
    for(const auto& path : read_only_paths)
    {
        if(path.empty())
            continue;
        auto ro = make_problem_cache_backend(path);
        ro.load(path);
        read_only_backends.push_back(std::move(ro));
    }
}

bool problem_cache::has(const std::string& name, const value& problem) const
{
    const auto key = create_key(name, problem);
    const auto in  = [&](const auto& b) { return b.has(device_key, key); };
    // Writable caches first, then the read-only layers (lowest priority).
    return std::any_of(writable_backends.begin(), writable_backends.end(), in) or
           std::any_of(read_only_backends.begin(), read_only_backends.end(), in);
}

void problem_cache::insert(const std::string& name, const value& problem, const value& solution)
{
    assert(not solution.is_null());
    // New solutions go to the primary writable cache (always present: a file
    // cache, or the in-memory json seeded at construction/load).
    writable_backends.front().insert(device_key, create_key(name, problem), solution);
}

void problem_cache::mark(const std::string& name, const value& problem)
{
    writable_backends.front().mark(device_key, create_key(name, problem));
}

optional<value> problem_cache::get(const std::string& name, const value& problem) const
{
    const auto key = create_key(name, problem);
    // Writable caches first (a locally tuned solution wins), then the read-only
    // layers in priority order (first hit wins among them).
    const auto search = [&](const std::vector<problem_cache_backend>& backends) -> optional<value> {
        const auto it = std::find_if(backends.begin(), backends.end(), [&](const auto& b) {
            return b.get(device_key, key).has_value();
        });
        if(it != backends.end())
            return it->get(device_key, key);
        return {};
    };
    if(auto sol = search(writable_backends))
        return sol;
    return search(read_only_backends);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
