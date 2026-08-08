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
#include <migraphx/gpu/context.hpp>
#include <algorithm>
#include <cassert>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

problem_cache::problem_cache() : backend(json_problem_cache{}) {}

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

void problem_cache::load(const std::string& path)
{
    if(path.empty())
        return;
    // Remember the path so save() writes back here; a missing file loads empty.
    path_override = path;
    backend.load(path);
}

void problem_cache::save() const
{
    // No writable path (read-only or unconfigured cache) means save does nothing.
    if(path_override.empty())
        return;
    backend.save(path_override);
}

void problem_cache::load(const std::vector<std::string>& paths)
{
    read_only_backends.clear();
    if(paths.empty())
        return;
    // A single file is the writable cache (new solutions save back to it).
    if(paths.size() == 1)
    {
        load(paths.front());
        return;
    }
    // Multiple files are a read-only priority list (highest first). Shipped
    // caches are immutable, so the writable `backend` stays empty and nothing
    // is written back; persisting to a writable local cache is a future item.
    for(const auto& path : paths)
    {
        problem_cache_backend ro(json_problem_cache{});
        if(not path.empty())
            ro.load(path);
        read_only_backends.push_back(std::move(ro));
    }
}

bool problem_cache::has(const std::string& name, const value& problem) const
{
    const auto key = create_key(name, problem);
    // Read-only layers first (highest priority), then the writable cache.
    return std::any_of(read_only_backends.begin(),
                       read_only_backends.end(),
                       [&](const auto& ro) { return ro.has(device_key, key); }) or
           backend.has(device_key, key);
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
    // Read-only layers first (highest priority), then the writable cache.
    const auto found =
        std::find_if(read_only_backends.begin(), read_only_backends.end(), [&](const auto& ro) {
            return ro.get(device_key, key).has_value();
        });
    if(found != read_only_backends.end())
        return found->get(device_key, key);
    return backend.get(device_key, key);
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
