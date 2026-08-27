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
 */
#ifndef MIGRAPHX_GUARD_GPU_BINARY_CACHE_HPP
#define MIGRAPHX_GUARD_GPU_BINARY_CACHE_HPP

#include <migraphx/gpu/config.hpp>
#include <migraphx/gpu/compiled_code.hpp>
#include <migraphx/env.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/reflect.hpp>
#include <migraphx/value.hpp>
#include <memory>
#include <string>
#include <unordered_map>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

MIGRAPHX_DECLARE_ENV_VAR(MIGRAPHX_BINARY_CACHE)

struct context;

/// How the cache behaves for one compile. Supplied through the target's backend options, with
/// the MIGRAPHX_BINARY_CACHE environment variable as the default for the directory.
struct binary_cache_settings
{
    /// Where entries are kept. Nothing is written to disk when this is empty, though results
    /// are still shared in memory.
    std::string path = string_value_of(MIGRAPHX_BINARY_CACHE{});
    /// Compile even when a result can be reused, and fail if the two disagree.
    bool verify = false;
};

/**
 * Compiled kernels, keyed by a string describing what the compiler was given.
 *
 * Results are held in memory for the life of the context and, when a cache directory is
 * configured, written to disk so later runs can reuse them. Things outside the key, such as the
 * compiler and the embedded kernel headers, are separated by the directory the entries live in.
 *
 * This is not thread safe, and deliberately so. Every key is known before any compile begins, so
 * the compile pass looks results up and stores them in serial passes on either side of its
 * parallel compiles, and nothing has to be guarded.
 */
struct MIGRAPHX_GPU_EXPORT binary_cache
{
    /// What gets written to disk for one compiled kernel. The op name, problem and solution are
    /// stored for offline inspection; only the key is checked when an entry is loaded.
    struct entry
    {
        std::string key     = {};
        std::string op_name = {};
        value problem       = {};
        value solution      = {};
        compiled_code code  = {};

        template <class Self, class F>
        static auto reflect(Self& self, F f)
        {
            return pack(f(self.key, "key"),
                        f(self.op_name, "op_name"),
                        f(self.problem, "problem"),
                        f(self.solution, "solution"),
                        f(self.code, "code"));
        }
    };

    /// Counts of what the cache did.
    struct stats
    {
        /// Served from memory, from an earlier compile or disk read in this process.
        std::size_t reused = 0;
        /// Served from the cache directory.
        std::size_t hits = 0;
        /// Not found, so the caller had to compile.
        std::size_t misses = 0;
        /// Written to the cache after being compiled.
        std::size_t compiled = 0;
    };

    explicit binary_cache(binary_cache_settings s = {}) : settings(std::move(s)) {}

    /// Look up a key, consulting memory first and then the cache directory.
    optional<compiled_code> get(const context& ctx, const std::string& key);

    /// Record a compiled result under its key.
    void insert(const context& ctx, entry e);

    /// True when reused results should be checked against a fresh compile.
    bool verify() const { return settings.verify; }

    const stats& get_stats() const { return counters; }

    /// Names the directory holding entries that this build can use: the entry format, the
    /// compiler, a digest of the embedded kernel headers, and the rocMLIR build. Empty when the
    /// compiler cannot be identified, in which case nothing is read from or written to disk,
    /// since entries from different compilers could not be told apart.
    static const std::string& version_dir();

    private:
    std::unordered_map<std::string, compiled_code> memo;
    binary_cache_settings settings;
    stats counters;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_GPU_BINARY_CACHE_HPP
