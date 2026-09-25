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
    /// Open the database at path, creating the schema if it is writable.
    ///
    /// A database that can only be read serves lookups and ignores stores; it is used as it
    /// stands, without creating the schema. Returns nullopt when the database cannot be opened
    /// at all or entries cannot be looked up in it, so an unusable database leaves the cache
    /// memory-only rather than raising an error.
    static optional<sqlite_binary_cache> open(const std::string& path);

    optional<std::vector<char>>
    load(const std::string& version, const std::string& device, const std::string& key_hash) const;
    void store(const std::string& version,
               const std::string& device,
               const std::string& key_hash,
               const binary_cache_entry& e,
               const std::vector<char>& blob) const;

    /// Open a transaction, so the stores that follow cost one commit rather than one each.
    void begin_batch();
    /// Commit the transaction begin_batch opened, or roll it back if the commit fails.
    void end_batch();

    private:
    sqlite db              = {};
    sqlite_stmt get_stmt   = {};
    sqlite_stmt store_stmt = {};
    /// Whether begin_batch opened a transaction that end_batch still has to close.
    bool in_batch = false;
};

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx

#endif // MIGRAPHX_GUARD_GPU_SQLITE_BINARY_CACHE_HPP
