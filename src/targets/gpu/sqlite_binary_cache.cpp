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
#include <migraphx/gpu/binary_cache_backend.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/json.hpp>
#include <migraphx/logger.hpp>
#include <migraphx/md5.hpp>
#include <migraphx/msgpack.hpp>
#include <migraphx/serialize.hpp>
#include <cassert>
#include <cstdint>
#include <exception>
#include <type_traits>
#include <utility>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

static_assert(std::is_constructible<binary_cache_backend, sqlite_binary_cache>{},
              "sqlite_binary_cache must satisfy the binary_cache_backend concept");

namespace {

// How long to wait for a lock held by another process before giving up. This is the entire
// cross-process strategy: whatever still fails degrades to a recompile.
constexpr int busy_timeout_ms = 5000;

// The table name carries the schema version, so an incompatible change is a new table that old
// binaries ignore rather than a migration. This is orthogonal to binary_cache_format, which
// versions how entries are serialized and reaches each row through the version column.
//
// Deliberately not WITHOUT ROWID, unlike the sibling table in sqlite_problem_cache: that clause
// stores the payload inside the index B-tree, which suits short JSON but not a whole serialized
// program fragment, which would spill into overflow chains hanging off the index.
//
// Rows are addressed by a hash of the key rather than the key itself, which is the whole compile
// source and would otherwise be stored a second time in the index. The full key is kept in its
// own column and checked on load, so a collision is a miss rather than a wrong kernel.
//
// The primary key leads with version so that dropping everything belonging to a superseded
// toolchain is a range scan rather than a full table scan. Point lookups bind all three and do
// not care about the order.
constexpr const char* schema_sql = R"__migraphx__(
CREATE TABLE IF NOT EXISTS cache_v1 (
  version   TEXT    NOT NULL,
  device    TEXT    NOT NULL,
  key_hash  TEXT    NOT NULL,
  key       TEXT    NOT NULL,
  op_name   TEXT    NOT NULL,
  problem   TEXT    NOT NULL,
  solution  TEXT    NOT NULL,
  code      BLOB    NOT NULL,
  timestamp INTEGER NOT NULL,
  PRIMARY KEY (version, device, key_hash)
);
)__migraphx__";

constexpr const char* get_sql = "SELECT key, op_name, problem, solution, code FROM cache_v1"
                                " WHERE version = ?1 AND device = ?2 AND key_hash = ?3;";

// Overwriting is the analog of the file backend's publish-by-rename: content is decided by the
// key, so the last writer wins with an equivalent row. An upsert, since INSERT OR REPLACE deletes
// first and would cascade away the kernel's statistics. The timestamp comes from the database so
// rows from different machines compare, for pruning by age.
constexpr const char* store_sql =
    "INSERT INTO cache_v1"
    " (version, device, key_hash, key, op_name, problem, solution, code, timestamp)"
    " VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, CAST(STRFTIME('%s','now') AS INTEGER))"
    " ON CONFLICT (version, device, key_hash) DO UPDATE SET"
    " key = excluded.key, op_name = excluded.op_name, problem = excluded.problem,"
    " solution = excluded.solution, code = excluded.code, timestamp = excluded.timestamp;";

// Usage statistics, created only when asked for. Each row references its cache_v1 row so pruning
// the cache prunes its statistics. Foreign keys are per connection and off by default, so this
// pragma runs on each open, outside a transaction since inside one it is ignored.
constexpr const char* stats_schema_sql = R"__migraphx__(
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS model_v1 (
  model_id      INTEGER PRIMARY KEY,
  name          TEXT    NOT NULL,
  program_hash  TEXT    NOT NULL,
  inputs        TEXT    NOT NULL,
  options       TEXT    NOT NULL,
  UNIQUE (name, program_hash)
);
CREATE TABLE IF NOT EXISTS model_kernel_v1 (
  model_id   INTEGER NOT NULL,
  version    TEXT    NOT NULL,
  device     TEXT    NOT NULL,
  key_hash   TEXT    NOT NULL,
  use_count  INTEGER NOT NULL,
  PRIMARY KEY (model_id, version, device, key_hash),
  FOREIGN KEY (model_id) REFERENCES model_v1(model_id) ON DELETE CASCADE,
  FOREIGN KEY (version, device, key_hash)
    REFERENCES cache_v1(version, device, key_hash) ON DELETE CASCADE
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS model_kernel_v1_kernel ON model_kernel_v1(version, device, key_hash);
CREATE TABLE IF NOT EXISTS kernel_use_v1 (
  version    TEXT    NOT NULL,
  device     TEXT    NOT NULL,
  key_hash   TEXT    NOT NULL,
  use_count  INTEGER NOT NULL DEFAULT 0,
  hit_count  INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (version, device, key_hash),
  FOREIGN KEY (version, device, key_hash)
    REFERENCES cache_v1(version, device, key_hash) ON DELETE CASCADE
) WITHOUT ROWID;
)__migraphx__";

// Updating on conflict rather than DO NOTHING is what lets RETURNING hand back a known model's
// id, and keeps its options current.
constexpr const char* model_sql =
    "INSERT INTO model_v1 (name, program_hash, inputs, options) VALUES (?1, ?2, ?3, ?4)"
    " ON CONFLICT (name, program_hash) DO UPDATE SET"
    " inputs = excluded.inputs, options = excluded.options"
    " RETURNING model_id;";

// A compile records every kernel of the model, so the previous list for this toolchain and device
// is dropped first. Otherwise a kernel the model stopped using, such as when tuning picks a
// different solution, would stay attributed to it.
constexpr const char* clear_model_kernels_sql =
    "DELETE FROM model_kernel_v1 WHERE model_id = ?1 AND version = ?2 AND device = ?3;";

// Selecting from cache_v1 skips a kernel whose store failed, rather than letting its foreign key
// fail the whole transaction.
constexpr const char* model_kernel_sql =
    "INSERT INTO model_kernel_v1 (model_id, version, device, key_hash, use_count)"
    " SELECT ?1, version, device, key_hash, ?5 FROM cache_v1"
    " WHERE version = ?2 AND device = ?3 AND key_hash = ?4"
    " ON CONFLICT (model_id, version, device, key_hash) DO UPDATE SET"
    " use_count = excluded.use_count;";

constexpr const char* use_sql =
    "INSERT INTO kernel_use_v1 (version, device, key_hash, use_count, hit_count)"
    " SELECT version, device, key_hash, ?4, ?5 FROM cache_v1"
    " WHERE version = ?1 AND device = ?2 AND key_hash = ?3"
    " ON CONFLICT (version, device, key_hash) DO UPDATE SET"
    " use_count = use_count + excluded.use_count,"
    " hit_count = hit_count + excluded.hit_count;";

// Stores come in a burst after each round of compiles, and outside a transaction every one of
// them is its own commit, each waiting for the disk. IMMEDIATE takes the write lock up front, so
// a busy database is found out here, once, rather than partway through the stores; readers are
// not blocked until the commit itself.
constexpr const char* begin_sql    = "BEGIN IMMEDIATE;";
constexpr const char* commit_sql   = "COMMIT;";
constexpr const char* rollback_sql = "ROLLBACK;";

/// Run f in one write transaction. A transaction left open would hold the write lock against
/// every other process, so a failure rolls it back before rethrowing.
template <class F>
void in_transaction(sqlite& db, const char* what, F f)
{
    db.execute(begin_sql);
    try
    {
        f();
        db.execute(commit_sql);
    }
    catch(...)
    {
        try
        {
            db.execute(rollback_sql);
        }
        catch(const std::exception& ex)
        {
            log::warn() << "Failed to roll back binary cache " << what << ": " << ex.what();
        }
        throw;
    }
}

} // namespace

optional<sqlite_binary_cache> sqlite_binary_cache::open(const std::string& path, bool stats)
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
            r.db.execute(schema_sql);
            r.store_stmt = r.db.prepare(store_sql);
            if(stats)
                r.open_stats(path);
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
    return r;
}

optional<binary_cache_entry> sqlite_binary_cache::load(const std::string& version,
                                                       const std::string& device,
                                                       const std::string& key) const
{
    auto key_hash = md5(key);
    try
    {
        // The primary key makes this at most one row.
        auto rows = get_stmt(version, device, key_hash);
        auto it   = rows.begin();
        if(it == rows.end())
            return nullopt;
        auto row = *it;
        // Rows are addressed by a hash of the key, so the full key is checked here to make a
        // collision a miss rather than a wrong kernel.
        if(row.at("key").get_string() != key)
        {
            log::warn() << "Ignoring binary cache entry with mismatched key: " << key_hash;
            return nullopt;
        }
        binary_cache_entry e;
        e.key            = key;
        e.op_name        = row.at("op_name").get_string();
        e.problem        = from_json_string(row.at("problem").get_string());
        e.solution       = from_json_string(row.at("solution").get_string());
        const auto& code = row.at("code").get_binary();
        migraphx::from_value(from_msgpack(reinterpret_cast<const char*>(code.data()), code.size()),
                             e.code);
        return e;
    }
    catch(const std::exception& ex)
    {
        // A cache that cannot be read, or a damaged row, is a miss, which costs a recompile and
        // nothing else.
        log::warn() << "Ignoring unreadable binary cache entry " << key_hash << ": " << ex.what();
        return nullopt;
    }
}

void sqlite_binary_cache::store(const std::string& version,
                                const std::string& device,
                                const std::vector<binary_cache_entry>& entries)
{
    // Never prepared for a read-only database, and cleared after a failed store; either way
    // stores are skipped.
    if(not store_stmt.valid())
        return;
    try
    {
        in_transaction(db, "entries", [&] {
            for(const auto& e : entries)
            {
                store_stmt(version,
                           device,
                           md5(e.key),
                           e.key,
                           e.op_name,
                           to_json_string(e.problem),
                           to_json_string(e.solution),
                           to_msgpack(migraphx::to_value(e.code)));
            }
        });
    }
    catch(...)
    {
        // A database that refused one write, often after waiting out the busy timeout, would
        // most likely refuse the next one as well, so later stores are skipped and the cache
        // carries on read-only. Lookups keep working.
        store_stmt = {};
        log::warn() << "Binary cache is read-only from now on";
        throw;
    }
}

void sqlite_binary_cache::open_stats(const std::string& path)
{
    try
    {
        db.execute(stats_schema_sql);
        model_stmt        = db.prepare(model_sql);
        clear_model_stmt  = db.prepare(clear_model_kernels_sql);
        model_kernel_stmt = db.prepare(model_kernel_sql);
        use_stmt          = db.prepare(use_sql);
    }
    catch(const std::exception& ex)
    {
        model_stmt        = {};
        clear_model_stmt  = {};
        model_kernel_stmt = {};
        use_stmt          = {};
        log::warn() << "Disabling binary cache statistics at " << path << ": " << ex.what();
    }
}

void sqlite_binary_cache::record_usage(const std::string& version,
                                       const std::string& device,
                                       const optional<binary_cache_model>& model,
                                       const binary_cache_usage_map& usage)
{
    if(not has_stats() or (usage.empty() and not model.has_value()))
        return;
    in_transaction(db, "statistics", [&] {
        std::int64_t model_id = 0;
        if(model.has_value())
        {
            auto rows = model_stmt(model->name,
                                   model->program_hash,
                                   to_json_string(model->inputs),
                                   to_json_string(model->options));
            // An upsert with RETURNING always yields its one row.
            auto it = rows.begin();
            assert(it != rows.end());
            model_id = (*it).at("model_id").to<std::int64_t>();
            clear_model_stmt(model_id, version, device);
        }
        for(const auto& [key, u] : usage)
        {
            auto key_hash = md5(key);
            use_stmt(version, device, key_hash, u.uses, u.hits);
            // A kernel only loaded, such as a tuning candidate that lost, is not part of the model.
            if(model.has_value() and u.uses > 0)
                model_kernel_stmt(model_id, version, device, key_hash, u.uses);
        }
    });
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
