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
#include <migraphx/gpu/json_problem_cache.hpp>
#include <migraphx/gpu/problem_cache_backend.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/json.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/logger.hpp>
#include <algorithm>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Compile-time confirmation that json_problem_cache satisfies the backend
// concept. If a method signature drifts, this assertion fires at the
// definition site rather than at some far-away usage.
static_assert(std::is_constructible<problem_cache_backend, json_problem_cache>{},
              "json_problem_cache must satisfy the problem_cache_backend concept");

static value create_key(const std::string& name, const value& problem)
{
    return {{"name", name}, {"problem", problem}};
}

void json_problem_cache::set_migration_device_key(cache_device_key key)
{
    migration_device_key = std::move(key);
}

void json_problem_cache::load(const std::string& path)
{
    if(path.empty())
        return;
    if(not fs::exists(path))
    {
        // Missing file is not an error: typical first-run case. Caller may
        // immediately follow with save() to create the file.
        return;
    }
    auto root = from_json_string(read_string(path));

    // Detect on-disk format.
    // - Array of [key, inner_map] pairs => current device-keyed format
    //   produced by problem_cache::save (and this class's save()).
    // - Object whose keys begin with '{' => legacy flat-object format
    //   (string-keyed by serialized {name, problem}). Migrated into the
    //   migration_device_key bucket. Sorting before insertion makes
    //   migration deterministic when duplicate keys collide.
    // - Anything else => unrecognised pre-merge format; skip rather than
    //   throw so a stale file does not block first-run.
    if(root.is_array())
    {
        // Canonicalize keys only (JSON erases value types); solutions verbatim.
        std::unordered_map<cache_device_key, std::unordered_map<value, value>> raw;
        from_value(root, raw);
        for(const auto& dev : raw)
            for(const auto& entry : dev.second)
                cache[dev.first][entry.first.normalize()] = entry.second;
        return;
    }
    if(not root.is_object() or root.empty())
        return;
    const auto& first_key = root.begin()->get_key();
    if(first_key.empty() or first_key.front() != '{')
    {
        log::warn() << "json_problem_cache: unrecognised on-disk format at " << path
                    << ", starting fresh";
        return;
    }

    std::unordered_map<value, value> flat;
    from_value(root, flat);
    std::vector<std::pair<value, value>> sorted_flat(flat.begin(), flat.end());
    std::sort(sorted_flat.begin(), sorted_flat.end(), [](const auto& a, const auto& b) {
        return to_json_string(a.first) < to_json_string(b.first);
    });
    auto& bucket = cache[migration_device_key];
    for(const auto& entry : sorted_flat)
    {
        auto projected =
            create_key(entry.first.at("name").to<std::string>(), entry.first.at("problem"));
        bucket.emplace(projected.normalize(), entry.second);
    }
    log::info() << "json_problem_cache: migrated " << flat.size()
                << " legacy entries into device bucket "
                << to_json_string(to_value(migration_device_key));
}

void json_problem_cache::save(const std::string& path) const
{
    if(path.empty())
        return;
    write_string(path, to_pretty_json_string(to_value(cache)));
}

void json_problem_cache::insert(const cache_device_key& dk, const value& key, const value& solution)
{
    assert(not solution.is_null());
    cache[dk][key.normalize()] = solution;
}

void json_problem_cache::mark(const cache_device_key& dk, const value& key)
{
    cache[dk].insert(std::make_pair(key.normalize(), value{}));
}

optional<value> json_problem_cache::get(const cache_device_key& dk, const value& key) const
{
    auto bucket_it = cache.find(dk);
    if(bucket_it == cache.end())
        return nullopt;
    auto it = bucket_it->second.find(key.normalize());
    if(it == bucket_it->second.end())
        return nullopt;
    return it->second;
}

bool json_problem_cache::has(const cache_device_key& dk, const value& key) const
{
    auto bucket_it = cache.find(dk);
    if(bucket_it == cache.end())
        return false;
    return contains(bucket_it->second, key.normalize());
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
