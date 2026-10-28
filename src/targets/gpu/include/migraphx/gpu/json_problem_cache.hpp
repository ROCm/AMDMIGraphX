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
#ifndef MIGRAPHX_GUARD_GPU_JSON_PROBLEM_CACHE_HPP
#define MIGRAPHX_GUARD_GPU_JSON_PROBLEM_CACHE_HPP

#include <migraphx/config.hpp>
#include <migraphx/value.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/gpu/export.h>
#include <migraphx/gpu/cache_device_key.hpp>

#include <string>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// A problem_cache_backend that persists entries as JSON, using the same
// on-disk format as problem_cache::save() (an array of [cache_device_key,
// inner_map] pairs). Path resolution is the caller's job.
struct MIGRAPHX_GPU_EXPORT json_problem_cache
{
    // problem_cache_backend concept members:
    void load(const std::string& path);
    void save(const std::string& path) const;
    void insert(const cache_device_key& dk, const value& key, const value& solution);
    void mark(const cache_device_key& dk, const value& key);
    optional<value> get(const cache_device_key& dk, const value& key) const;
    bool has(const cache_device_key& dk, const value& key) const;

    // Device bucket -> ({name, problem} -> solution).
    std::unordered_map<cache_device_key, std::unordered_map<value, value>> cache;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_JSON_PROBLEM_CACHE_HPP
