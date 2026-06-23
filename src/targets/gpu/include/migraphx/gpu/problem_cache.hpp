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
#include <migraphx/hash.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/gpu/export.h>
#include <cstddef>
#include <string>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Identifies the device a problem_cache bucket was populated for. Owning
// gpu::context builds this from its already-captured hip_device properties --
// problem_cache must not query the HIP runtime itself, so AOT cross-compilation
// (where the host GPU differs from the target GPU) is correctly keyed by the
// target's properties rather than the host's.
struct cache_device_key
{
    std::string device_name    = {};
    std::string gfx_name       = {};
    std::size_t cu_count       = 0;
    std::size_t wavefront_size = 0;

    template <class Self, class F>
    static auto reflect(Self& self, F f)
    {
        return pack(f(self.device_name, "device_name"),
                    f(self.gfx_name, "gfx_name"),
                    f(self.cu_count, "cu_count"),
                    f(self.wavefront_size, "wavefront_size"));
    }

    friend bool operator==(const cache_device_key& a, const cache_device_key& b)
    {
        return a.device_name == b.device_name and a.gfx_name == b.gfx_name and
               a.cu_count == b.cu_count and a.wavefront_size == b.wavefront_size;
    }
    friend bool operator!=(const cache_device_key& a, const cache_device_key& b)
    {
        return not(a == b);
    }
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

namespace std {
template <>
struct hash<migraphx::gpu::cache_device_key>
{
    std::size_t operator()(const migraphx::gpu::cache_device_key& k) const noexcept
    {
        std::size_t seed = 0;
        migraphx::hash_combine(seed, k.device_name);
        migraphx::hash_combine(seed, k.gfx_name);
        migraphx::hash_combine(seed, k.cu_count);
        migraphx::hash_combine(seed, k.wavefront_size);
        return seed;
    }
};
} // namespace std

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

struct MIGRAPHX_GPU_EXPORT problem_cache
{
    void set_device_key(cache_device_key key);
    const cache_device_key& get_device_key() const;

    bool has(const std::string& name, const value& problem) const;
    void insert(const std::string& name, const value& problem, const value& solution);
    void mark(const std::string& name, const value& problem);
    optional<value> get(const std::string& name, const value& problem) const;
    void load();
    void save() const;
    // Outer key identifies the device whose properties produced the entries
    // (device name, gfx name, CU count, wavefront size). Each device has its
    // own inner map of {name, problem} -> solution. This structure lets a
    // single cache file hold solutions for multiple GPU configurations
    // without collisions; lookups via has/get/insert/mark implicitly key into
    // the current device's bucket only.
    std::unordered_map<cache_device_key, std::unordered_map<value, value>> cache;

    private:
    // Stable identifier for the device whose properties this cache is bound
    // to. Set by the owning context from hip_device::get_device_key (see
    // gpu/context.hpp). A default-constructed key (empty strings, zero
    // counts) means the device couldn't be identified -- entries land in a
    // single anonymous bucket, which is still deterministic but not labeled.
    cache_device_key device_key{};
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_PROBLEM_CACHE_HPP
