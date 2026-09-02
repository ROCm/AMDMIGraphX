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
#include <migraphx/gpu/sqlite_binary_cache.hpp>
#include <migraphx/gpu/binary_cache.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/json.hpp>
#include <migraphx/logger.hpp>
#include <type_traits>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Compile-time confirmation that sqlite_binary_cache satisfies the backend concept. If a method
// signature drifts, this assertion fires at the definition site rather than at some far-away
// usage.
static_assert(std::is_constructible<binary_cache_backend, sqlite_binary_cache>{},
              "sqlite_binary_cache must satisfy the binary_cache_backend concept");

namespace {

// How long to wait for a lock held by another process before giving up, matching rocFFT. This
// is the entire cross-process strategy: whatever still fails degrades to a recompile.
constexpr int busy_timeout_ms = 5000;

// The table name carries the schema version, so an incompatible change is a new table that old
// binaries ignore rather than a migration. This is orthogonal to binary_cache_format, which
// versions the entry payload and reaches the row through the version column.
//
// Deliberately not WITHOUT ROWID, unlike the sibling table in sqlite_problem_cache: that clause
// stores the payload inside the index B-tree, which suits short JSON but not a whole serialized
// program fragment, which would spill into overflow chains hanging off the index.
//
// The primary key leads with version so that dropping everything belonging to a superseded
// toolchain is a range scan rather than a full table scan. Point lookups bind all three and do
// not care about the order.
constexpr const char* schema_sql = R"__migraphx__(
CREATE TABLE IF NOT EXISTS cache_v1 (
  version   TEXT    NOT NULL,
  device    TEXT    NOT NULL,
  key_hash  TEXT    NOT NULL,
  op_name   TEXT    NOT NULL,
  problem   TEXT    NOT NULL,
  solution  TEXT    NOT NULL,
  entry     BLOB    NOT NULL,
  timestamp INTEGER NOT NULL,
  PRIMARY KEY (version, device, key_hash)
);
CREATE TABLE IF NOT EXISTS cache_info_v1 (
  version TEXT PRIMARY KEY,
  stamp   TEXT NOT NULL
);
)__migraphx__";

constexpr const char* get_sql =
    "SELECT entry FROM cache_v1 WHERE version = ?1 AND device = ?2 AND key_hash = ?3;";

// INSERT OR REPLACE is the analogue of the file backend's publish-by-rename: the content is
// decided entirely by the key, so two processes compiling the same kernel is benign and the
// last writer wins with equivalent bytes. The timestamp is computed by the database so it is
// consistent across writers.
constexpr const char* store_sql =
    "INSERT OR REPLACE INTO cache_v1"
    " (version, device, key_hash, op_name, problem, solution, entry, timestamp)"
    " VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, CAST(STRFTIME('%s','now') AS INTEGER));";

constexpr const char* info_sql =
    "INSERT OR IGNORE INTO cache_info_v1 (version, stamp) VALUES (?1, ?2);";

} // namespace

optional<binary_cache_backend> sqlite_binary_cache::open(const std::string& path)
{
    sqlite_binary_cache r;
    try
    {
        // sqlite will not create missing directories, but the file backend does, so without
        // this a fresh machine would silently get no cache from a path that would have worked
        // had it named a directory instead of a database.
        auto parent = fs::path{path}.parent_path();
        if(not parent.empty())
            fs::create_directories(parent);
        auto db = sqlite::try_write(path);
        if(not db.has_value())
        {
            log::warn() << "Disabling the binary cache: cannot open " << path;
            return nullopt;
        }
        r.db = std::move(*db);
        r.db.set_busy_timeout(busy_timeout_ms);
        (void)r.db.execute(schema_sql);
        // Without a working lookup there is no cache, so this failure disables the backend.
        r.get_stmt = r.db.prepare(get_sql);
    }
    catch(const std::exception& ex)
    {
        log::warn() << "Disabling the binary cache at " << path << ": " << ex.what();
        return nullopt;
    }
    try
    {
        r.store_stmt = r.db.prepare(store_sql);
        r.info_stmt  = r.db.prepare(info_sql);
    }
    catch(const std::exception& ex)
    {
        // A database that can be read but not written to is still worth having: reads serve
        // hits and stores quietly do nothing.
        log::warn() << "Binary cache at " << path << " is read-only: " << ex.what();
    }
    return binary_cache_backend{std::move(r)};
}

void sqlite_binary_cache::stamp_version(const std::string& version)
{
    if(info_written == version or not info_stmt.valid())
        return;
    try
    {
        sqlite_stmt_reset guard{info_stmt};
        info_stmt.bind(1, version).bind(2, binary_cache::version_stamp());
        info_stmt.step();
        info_written = version;
    }
    catch(const std::exception& ex)
    {
        // The stamp is provenance for a human reading the database later, so failing to write
        // it must not stop entries being stored. Remember the version anyway so a database
        // that rejects this never retries it once per kernel.
        log::warn() << "Failed to stamp the binary cache: " << ex.what();
        info_written = version;
    }
}

optional<std::vector<char>> sqlite_binary_cache::load(const std::string& version,
                                                      const std::string& device,
                                                      const std::string& key_hash)
{
    if(not get_stmt.valid())
        return nullopt;
    try
    {
        sqlite_stmt_reset guard{get_stmt};
        get_stmt.bind(1, version).bind(2, device).bind(3, key_hash);
        if(not get_stmt.step())
            return nullopt;
        return get_stmt.column_blob(0);
    }
    catch(const std::exception& ex)
    {
        // A cache that cannot be read is a miss, which costs a recompile and nothing else.
        log::warn() << "Failed to read binary cache entry " << key_hash << ": " << ex.what();
        return nullopt;
    }
}

void sqlite_binary_cache::store(const std::string& version,
                                const std::string& device,
                                const std::string& key_hash,
                                const binary_cache_entry& e,
                                const std::vector<char>& blob)
{
    if(not store_stmt.valid())
        return;
    stamp_version(version);
    try
    {
        // The json strings are temporaries, which is safe because binding copies immediately.
        sqlite_stmt_reset guard{store_stmt};
        store_stmt.bind(1, version)
            .bind(2, device)
            .bind(3, key_hash)
            .bind(4, e.op_name)
            .bind(5, to_json_string(e.problem))
            .bind(6, to_json_string(e.solution))
            .bind(7, blob);
        store_stmt.step();
    }
    catch(const std::exception& ex)
    {
        log::warn() << "Failed to write binary cache entry " << key_hash << ": " << ex.what();
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
