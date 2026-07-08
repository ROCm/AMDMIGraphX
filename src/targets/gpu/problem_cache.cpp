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
#include <migraphx/gpu/context.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/json.hpp>
#include <migraphx/env.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/file_buffer.hpp>
#include <migraphx/logger.hpp>
#include <algorithm>
#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_PROBLEM_CACHE)

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

const cache_device_key& problem_cache::get_device_key() const { return device_key; }

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
    auto root = from_json_string(read_string(pc_path)).normalize();

    // Detect on-disk format. The new device-keyed format serializes the outer
    // map (unordered_map<cache_device_key, ...>) as a JSON array of
    // [key_object, inner_map] pairs. The legacy flat format used a JSON
    // object whose keys were JSON-encoded {name, problem} strings (i.e. they
    // begin with '{'). Anything else is an unrecognized pre-merge format
    // (e.g. an earlier prototype that string-keyed by hardware fingerprint);
    // skip rather than throw, so users with stale files start fresh.
    if(root.is_array())
    {
        from_value(root, cache);
        return;
    }
    if(not root.is_object() or root.empty())
        return;
    const auto& first_key = root.begin()->get_key();
    if(first_key.empty() or first_key.front() != '{')
    {
        log::warn() << "Problem cache: unrecognized on-disk format, starting fresh";
        return;
    }

    // Legacy flat-object format: migrate entries into the current device's
    // bucket. Sort by serialized key first so deduplication is deterministic
    // when multiple entries project to the same {name, problem}.
    std::unordered_map<value, value> flat;
    from_value(root, flat);
    std::vector<std::pair<value, value>> sorted_flat(flat.begin(), flat.end());
    std::sort(sorted_flat.begin(), sorted_flat.end(), [](const auto& a, const auto& b) {
        return to_json_string(a.first) < to_json_string(b.first);
    });
    auto& bucket = cache[device_key];
    for(const auto& entry : sorted_flat)
    {
        auto projected =
            create_key(entry.first.at("name").to<std::string>(), entry.first.at("problem"));
        bucket.emplace(projected, entry.second);
    }
    log::info() << "Problem cache: migrated " << flat.size()
                << " legacy entries into device bucket " << to_json_string(to_value(device_key));
}

void problem_cache::save() const
{
    auto pc_path = string_value_of(MIGRAPHX_PROBLEM_CACHE{});
    if(pc_path.empty())
        return;
    write_string(pc_path, to_pretty_json_string(to_value(cache)));
}

bool problem_cache::has(const std::string& name, const value& problem) const
{
    auto bucket_it = cache.find(device_key);
    if(bucket_it == cache.end())
        return false;
    return contains(bucket_it->second, create_key(name, problem));
}

void problem_cache::insert(const std::string& name, const value& problem, const value& solution)
{
    assert(not solution.is_null());
    cache[device_key][create_key(name, problem)] = solution;
}

void problem_cache::mark(const std::string& name, const value& problem)
{
    cache[device_key].insert(std::make_pair(create_key(name, problem), value{}));
}

optional<value> problem_cache::get(const std::string& name, const value& problem) const
{
    auto bucket_it = cache.find(device_key);
    if(bucket_it == cache.end())
        return nullopt;
    auto it = bucket_it->second.find(create_key(name, problem));
    if(it == bucket_it->second.end())
        return nullopt;
    return it->second;
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
