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

// The default backend is JSON for an unconfigured/in-memory cache; load() picks
// the backend by file type once a path is known.
problem_cache::problem_cache() : backend(json_problem_cache{}) {}

// Select the storage backend by file type: a ".db"/".sqlite" path uses the
// SQLite backend, anything else (including no extension) uses JSON. This lets a
// priority list mix cache formats.
static problem_cache_backend make_backend(const std::string& path)
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
    // No writable path (read-only or unconfigured cache) means save does nothing.
    if(path_override.empty())
        return;
    backend.save(path_override);
}

void problem_cache::load(const std::vector<std::string>& read_only_paths,
                         const std::vector<std::string>& writable_paths)
{
    read_only_backends.clear();
    path_override.clear();
    // Reset the writable backend; only a writable path below re-populates it.
    backend = problem_cache_backend{json_problem_cache{}};

    // Read/write cache (the developer cache): new solutions save back here. Only
    // one file is written, so the first non-empty writable path wins. Pick the
    // backend by file type and remember the path so save() writes back here; a
    // missing file loads empty.
    for(const auto& path : writable_paths)
    {
        if(not path.empty())
        {
            backend       = make_backend(path);
            path_override = path;
            backend.load(path);
            break;
        }
    }

    // Read-only caches (e.g. shipped by gpuep or an ISV): searched after the
    // writable cache and never written back.
    for(const auto& path : read_only_paths)
    {
        problem_cache_backend ro = make_backend(path);
        if(not path.empty())
            ro.load(path);
        read_only_backends.push_back(std::move(ro));
    }
}

bool problem_cache::has(const std::string& name, const value& problem) const
{
    const auto key = create_key(name, problem);
    // Writable cache first, then the read-only layers (lowest priority).
    return backend.has(device_key, key) or
           std::any_of(read_only_backends.begin(), read_only_backends.end(), [&](const auto& ro) {
               return ro.has(device_key, key);
           });
}

void problem_cache::insert(const std::string& name, const value& problem, const value& solution)
{
    assert(not solution.is_null());
    backend.insert(device_key, create_key(name, problem), solution);
}

void problem_cache::mark(const std::string& name, const value& problem)
{
    backend.mark(device_key, create_key(name, problem));
}

optional<value> problem_cache::get(const std::string& name, const value& problem) const
{
    const auto key = create_key(name, problem);
    // Writable cache first (a locally tuned solution wins), then the read-only
    // layers in priority order (first hit wins among them).
    if(auto sol = backend.get(device_key, key))
        return sol;
    const auto found =
        std::find_if(read_only_backends.begin(), read_only_backends.end(), [&](const auto& ro) {
            return ro.get(device_key, key).has_value();
        });
    if(found != read_only_backends.end())
        return found->get(device_key, key);
    return {};
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
