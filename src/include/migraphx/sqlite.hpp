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
#ifndef MIGRAPHX_GUARD_MIGRAPHX_SQLITE_HPP
#define MIGRAPHX_GUARD_MIGRAPHX_SQLITE_HPP

#include <migraphx/config.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/optional.hpp>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

struct sqlite_impl;
struct sqlite_stmt_impl;

/// A prepared statement, holding a reference to the connection it was prepared on so it can
/// never outlive it.
///
/// Not thread safe: one statement may be used by one thread at a time, even though the
/// connection itself is serialized. Reuse is the point of preparing -- prepare once, then
/// reset/bind/step per operation.
struct MIGRAPHX_EXPORT sqlite_stmt
{
    sqlite_stmt() = default;

    /// Bind a parameter. Indices are 1-based, matching sqlite's own convention.
    sqlite_stmt& bind(int i, std::string_view s);
    sqlite_stmt& bind(int i, std::int64_t x);
    sqlite_stmt& bind(int i, const std::vector<char>& blob);

    /// Step once. True when a row is available, false when the statement is done.
    bool step();

    /// Clear bindings and rewind, so the statement can be used again. Safe at any point,
    /// including after step() has thrown.
    void reset() noexcept;

    /// Read a column of the current row. Indices here are 0-based, again matching sqlite.
    std::string column_text(int i) const;
    std::vector<char> column_blob(int i) const;

    bool valid() const { return impl != nullptr; }

    private:
    friend struct sqlite;
    std::shared_ptr<sqlite_stmt_impl> impl;
};

/// Resets a statement on scope exit, so an early return or a thrown exception cannot leave
/// bindings or a half-consumed result set behind for whoever uses the statement next.
struct sqlite_stmt_reset
{
    explicit sqlite_stmt_reset(sqlite_stmt& s) : stmt(&s) {}
    sqlite_stmt_reset(const sqlite_stmt_reset&)            = delete;
    sqlite_stmt_reset& operator=(const sqlite_stmt_reset&) = delete;
    ~sqlite_stmt_reset() { stmt->reset(); }

    private:
    sqlite_stmt* stmt;
};

struct MIGRAPHX_EXPORT sqlite
{
    sqlite() = default;
    static sqlite read(const fs::path& p);
    static sqlite write(const fs::path& p);

    /// Open for writing, or nullopt if the file cannot be opened or created. For callers
    /// that treat an unusable database as "no cache" rather than as an error.
    static optional<sqlite> try_write(const fs::path& p);

    std::vector<std::unordered_map<std::string, std::string>> execute(const std::string& s);

    sqlite_stmt prepare(const std::string& sql);

    /// How long to wait for a lock held by another connection before failing.
    void set_busy_timeout(int ms);

    bool valid() const { return impl != nullptr; }

    private:
    std::shared_ptr<sqlite_impl> impl;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_SQLITE_HPP
