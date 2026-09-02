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
#ifndef MIGRAPHX_GUARD_GPU_SQLITE_BINARY_CACHE_HPP
#define MIGRAPHX_GUARD_GPU_SQLITE_BINARY_CACHE_HPP

#include <migraphx/gpu/config.hpp>
#include <migraphx/gpu/binary_cache_entry.hpp>
#include <migraphx/gpu/binary_cache_backend.hpp>
#include <migraphx/sqlite.hpp>
#include <migraphx/optional.hpp>
#include <string>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// A binary_cache_backend that keeps entries as rows in a SQLite database, one row per
// (version, device, key_hash). The stored blob is byte-identical to what the file backend
// writes into a .mxr file, so the two are interchangeable payloads; op_name, problem and
// solution are additionally denormalized into columns so a cache can be inspected with SQL.
//
// Holding sqlite and sqlite_stmt by value does not leak the SQLite dependency into this
// target: migraphx/sqlite.hpp forward-declares both impl types and never includes sqlite3.h.
struct MIGRAPHX_GPU_EXPORT sqlite_binary_cache
{
    /// Open the database, create the schema and prepare the statements. `stamp` is the text
    /// recorded in cache_info_v1 to describe the build. Returns nullopt when any of that fails,
    /// so an unusable database leaves the cache memory-only rather than raising an error.
    /// Returns the wrapper so the caller can hand the result straight back.
    static optional<binary_cache_backend> open(const std::string& path, std::string stamp);

    optional<std::vector<char>>
    load(const std::string& version, const std::string& device, const std::string& key_hash);
    void store(const std::string& version,
               const std::string& device,
               const std::string& key_hash,
               const binary_cache_entry& e,
               const std::vector<char>& blob);

    private:
    /// Record what this build is, once per version per process, so a table full of hashes can
    /// be identified later. The analogue of the file backend's cache.info.
    void stamp_version(const std::string& version);

    sqlite db              = {};
    sqlite_stmt get_stmt   = {};
    sqlite_stmt store_stmt = {};
    sqlite_stmt info_stmt  = {};
    /// What to write into cache_info_v1; supplied by the caller.
    std::string stamp = {};
    /// The last version stamped, so the stamp costs one statement per process rather than one
    /// per stored kernel.
    std::string info_written = {};
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_SQLITE_BINARY_CACHE_HPP
