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
#include <migraphx/filesystem.hpp>
#include <migraphx/json.hpp>
#include <migraphx/logger.hpp>
#include <cassert>
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

// How long to wait for a lock held by another process before giving up. This is the entire
// cross-process strategy: whatever still fails degrades to a recompile.
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
)__migraphx__";

constexpr const char* get_sql =
    "SELECT entry FROM cache_v1 WHERE version = ?1 AND device = ?2 AND key_hash = ?3;";

// INSERT OR REPLACE is the analogue of the file backend's publish-by-rename: the content is
// decided entirely by the key, so two processes compiling the same kernel is benign and the
// last writer wins with equivalent bytes. The timestamp is computed by the database rather
// than the process so that rows written by different machines stay comparable; nothing reads
// it yet, it is there to make pruning an old cache by age possible.
constexpr const char* store_sql =
    "INSERT OR REPLACE INTO cache_v1"
    " (version, device, key_hash, op_name, problem, solution, entry, timestamp)"
    " VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, CAST(STRFTIME('%s','now') AS INTEGER));";

// Stores come in a burst after each round of compiles, and outside a transaction every one of
// them is its own commit, each waiting for the disk. IMMEDIATE takes the write lock up front, so
// a busy database is found out here, once, rather than partway through the stores; readers are
// not blocked until the commit itself.
constexpr const char* begin_sql    = "BEGIN IMMEDIATE;";
constexpr const char* commit_sql   = "COMMIT;";
constexpr const char* rollback_sql = "ROLLBACK;";

} // namespace

optional<binary_cache_backend> sqlite_binary_cache::open(const std::string& path)
{
    sqlite_binary_cache r;
    try
    {
        // sqlite will not create a missing parent directory, but the file backend does, so
        // this keeps the two backends behaving the same on a fresh machine. A failure here is
        // left to the open below, since an existing database may still be readable.
        auto parent = fs::path{path}.parent_path();
        std::error_code ec;
        if(not parent.empty())
            fs::create_directories(parent, ec);

        // A database that can be read but not written to is still worth having: reads serve
        // hits and nothing is stored. Opening for writing already falls back to reading when the
        // file itself is write-protected; reading is tried here for anything else that refuses
        // a writer, such as a read-only mount, as long as there is a database to read.
        auto db = sqlite::try_write(path);
        if(not db.has_value() and fs::exists(path))
            db = sqlite::read(path);
        if(not db.has_value())
        {
            log::warn() << "Disabling the binary cache: cannot open " << path;
            return nullopt;
        }
        r.db = std::move(*db);
        r.db.set_busy_timeout(busy_timeout_ms);
        if(r.db.read_only())
        {
            log::warn() << "Binary cache at " << path << " is read-only";
        }
        else
        {
            (void)r.db.execute(schema_sql);
            r.store_stmt    = r.db.prepare(store_sql);
            r.begin_stmt    = r.db.prepare(begin_sql);
            r.commit_stmt   = r.db.prepare(commit_sql);
            r.rollback_stmt = r.db.prepare(rollback_sql);
        }
        // Without a working lookup there is no cache, so this failure disables the backend. That
        // includes a read-only database that was never given the schema.
        r.get_stmt = r.db.prepare(get_sql);
    }
    catch(const std::exception& ex)
    {
        log::warn() << "Disabling the binary cache at " << path << ": " << ex.what();
        return nullopt;
    }
    return binary_cache_backend{std::move(r)};
}

optional<std::vector<char>> sqlite_binary_cache::load(const std::string& version,
                                                      const std::string& device,
                                                      const std::string& key_hash) const
{
    if(not get_stmt.valid())
        return nullopt;
    try
    {
        // The primary key makes this at most one row.
        auto rows = get_stmt(version, device, key_hash);
        auto it   = rows.begin();
        if(it == rows.end())
            return nullopt;
        auto row          = *it;
        const auto& entry = row.at("entry").get_binary();
        return std::vector<char>(entry.begin(), entry.end());
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
                                const std::vector<char>& blob) const
{
    if(not store_stmt.valid())
        return;
    try
    {
        // The json strings are temporaries, which is safe because binding copies immediately.
        // An insert returns no rows, and it has run by the time the call returns.
        store_stmt(version,
                   device,
                   key_hash,
                   e.op_name,
                   to_json_string(e.problem),
                   to_json_string(e.solution),
                   blob);
    }
    catch(const std::exception& ex)
    {
        log::warn() << "Failed to write binary cache entry " << key_hash << ": " << ex.what();
    }
}

void sqlite_binary_cache::begin_batch()
{
    assert(not in_batch);
    if(not begin_stmt.valid())
        return;
    try
    {
        begin_stmt();
        in_batch = true;
    }
    catch(const std::exception& ex)
    {
        // Without a transaction each store commits on its own, which is slower but still works.
        log::warn() << "Binary cache stores will be committed one at a time: " << ex.what();
    }
}

void sqlite_binary_cache::end_batch()
{
    if(not in_batch)
        return;
    in_batch = false;
    try
    {
        commit_stmt();
    }
    catch(const std::exception& ex)
    {
        // A commit that fails leaves the transaction open, holding the write lock against every
        // other process, so it is rolled back and the batch's entries are lost instead.
        log::warn() << "Failed to commit binary cache entries: " << ex.what();
        try
        {
            rollback_stmt();
        }
        catch(const std::exception& rex)
        {
            log::warn() << "Failed to roll back binary cache entries: " << rex.what();
        }
    }
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
