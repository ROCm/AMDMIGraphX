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
#include <migraphx/gpu/sqlite_problem_cache.hpp>
#include <migraphx/gpu/problem_cache_backend.hpp>
#include <migraphx/sqlite.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/json.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/logger.hpp>

#include <utility>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {
namespace gpu {

// Compile-time confirmation that sqlite_problem_cache satisfies the backend
// concept. If a method signature drifts, this assertion fires at the
// definition site rather than at some far-away usage.
static_assert(std::is_constructible<problem_cache_backend, sqlite_problem_cache>{},
              "sqlite_problem_cache must satisfy the problem_cache_backend concept");

namespace {

constexpr const char* schema_sql = "CREATE TABLE IF NOT EXISTS solutions ("
                                   "  device_key  TEXT NOT NULL,"
                                   "  problem_key TEXT NOT NULL,"
                                   "  solution    TEXT NOT NULL,"
                                   "  PRIMARY KEY (device_key, problem_key)"
                                   ") WITHOUT ROWID;";

// Quote a string as a SQL literal, escaping embedded single quotes. Needed
// because migraphx::sqlite executes raw SQL text (no parameter binding).
std::string sql_quote(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 2);
    out += '\'';
    for(char c : s)
    {
        if(c == '\'')
            out += "''";
        else
            out += c;
    }
    out += '\'';
    return out;
}

} // namespace

void sqlite_problem_cache::load(const std::string& path)
{
    // Missing file is not an error: typical first-run case.
    if(path.empty() or not fs::exists(path))
        return;

    auto db = sqlite::read(path);
    std::vector<std::unordered_map<std::string, std::string>> rows;
    try
    {
        rows = db.execute("SELECT device_key, problem_key, solution FROM solutions;");
    }
    catch(const std::exception& e)
    {
        log::warn() << "sqlite_problem_cache: cannot read solutions from " << path << ": "
                    << e.what();
        return;
    }
    for(const auto& row : rows)
    {
        cache_device_key dk;
        from_value(from_json_string(row.at("device_key")), dk);
        // Normalize keys on load: JSON erases value types, so a serialized key
        // must be canonicalized to match the normalized runtime lookup key.
        cache[dk][from_json_string(row.at("problem_key")).normalize()] =
            from_json_string(row.at("solution"));
    }
}

void sqlite_problem_cache::save(const std::string& path) const
{
    if(path.empty())
        return;

    // Rewrite the table in one transaction so the file is deterministic.
    auto db         = sqlite::write(path);
    std::string sql = "BEGIN IMMEDIATE;";
    sql += schema_sql;
    sql += "DELETE FROM solutions;";
    for(const auto& bucket : cache)
    {
        const std::string dk = to_json_string(to_value(bucket.first));
        for(const auto& kv : bucket.second)
        {
            sql += "INSERT OR REPLACE INTO solutions(device_key, problem_key, solution) VALUES(";
            sql += sql_quote(dk) + "," + sql_quote(to_json_string(kv.first)) + "," +
                   sql_quote(to_json_string(kv.second)) + ");";
        }
    }
    sql += "COMMIT;";
    (void)db.execute(sql);
}

void sqlite_problem_cache::insert(const cache_device_key& dk,
                                  const value& key,
                                  const value& solution)
{
    assert(not solution.is_null());
    cache[dk][key.normalize()] = solution;
}

void sqlite_problem_cache::mark(const cache_device_key& dk, const value& key)
{
    cache[dk].insert(std::make_pair(key.normalize(), value{}));
}

optional<value> sqlite_problem_cache::get(const cache_device_key& dk, const value& key) const
{
    auto bucket_it = cache.find(dk);
    if(bucket_it == cache.end())
        return nullopt;
    auto it = bucket_it->second.find(key.normalize());
    if(it == bucket_it->second.end())
        return nullopt;
    return it->second;
}

bool sqlite_problem_cache::has(const cache_device_key& dk, const value& key) const
{
    auto bucket_it = cache.find(dk);
    if(bucket_it == cache.end())
        return false;
    return contains(bucket_it->second, key.normalize());
}

} // namespace gpu
} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
