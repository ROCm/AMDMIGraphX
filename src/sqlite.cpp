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

#include <migraphx/sqlite.hpp>
#include <migraphx/manage_ptr.hpp>
#include <migraphx/errors.hpp>
#include <migraphx/ranges.hpp>
#include <sqlite3.h>
#include <algorithm>
#include <cassert>
#include <iterator>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using sqlite3_ptr = MIGRAPHX_MANAGE_PTR(sqlite3*, sqlite3_close);

struct sqlite_impl
{
    sqlite3* get() const { return ptr.get(); }

    // sqlite3_open_v2 returns a handle even on failure (it carries the error message), so ptr
    // takes ownership either way.
    bool try_open(const fs::path& p, int flags)
    {
        sqlite3* ptr_tmp = nullptr;
        int rc           = sqlite3_open_v2(p.string().c_str(), &ptr_tmp, flags, nullptr);
        ptr              = sqlite3_ptr{ptr_tmp};
        return rc == 0;
    }

    void open(const fs::path& p, int flags)
    {
        if(not try_open(p, flags))
            MIGRAPHX_THROW("error opening " + p.string() + ": " + error_message());
    }

    template <class F>
    void exec(const char* sql, F f)
    {
        // cppcheck-suppress constParameterPointer
        auto callback = [](void* obj, auto... xs) -> int {
            try
            {
                const auto* g = static_cast<const F*>(obj);
                (*g)(xs...);
                return 0;
            }
            catch(...)
            {
                return -1;
            }
        };
        int rc = sqlite3_exec(get(), sql, callback, &f, nullptr);
        if(rc != 0)
            MIGRAPHX_THROW(error_message());
    }

    std::string error_message() const
    {
        std::string msg = "sqlite3: ";
        return msg + sqlite3_errmsg(get());
    }
    sqlite3_ptr ptr;
};

using sqlite3_stmt_ptr = MIGRAPHX_MANAGE_PTR(sqlite3_stmt*, sqlite3_finalize);

struct sqlite_stmt_impl
{
    sqlite3_stmt* get() const { return ptr.get(); }
    std::string error_message() const { return db->error_message(); }

    // Holding the connection keeps it alive while any statement on it exists. ptr is declared
    // after db so it is finalized first; finalizing after the connection closes is undefined.
    std::shared_ptr<sqlite_impl> db;
    sqlite3_stmt_ptr ptr;
};

constexpr int write_flags = SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE;

sqlite sqlite::read(const fs::path& p)
{
    sqlite r;
    r.impl = std::make_shared<sqlite_impl>();
    r.impl->open(p, SQLITE_OPEN_READONLY);
    return r;
}

sqlite sqlite::write(const fs::path& p)
{
    sqlite r;
    r.impl = std::make_shared<sqlite_impl>();
    r.impl->open(p, write_flags);
    return r;
}

optional<sqlite> sqlite::try_write(const fs::path& p)
{
    sqlite r;
    r.impl = std::make_shared<sqlite_impl>();
    if(not r.impl->try_open(p, write_flags))
        return nullopt;
    return r;
}

std::vector<std::unordered_map<std::string, std::string>> sqlite::execute(const std::string& s)
{
    std::vector<std::unordered_map<std::string, std::string>> result;
    impl->exec(s.c_str(), [&](int n, char** texts, char** names) {
        std::unordered_map<std::string, std::string> row;
        row.reserve(n);
        std::transform(
            names,
            names + n,
            texts,
            std::inserter(row, row.begin()),
            [&](const char* name, const char* text) { return std::make_pair(name, text); });
        result.push_back(row);
    });
    return result;
}

sqlite_stmt sqlite::prepare(const std::string& sql)
{
    sqlite3_stmt* stmt_tmp = nullptr;
    int rc                 = sqlite3_prepare_v2(impl->get(), sql.c_str(), -1, &stmt_tmp, nullptr);
    sqlite_stmt result;
    result.impl      = std::make_shared<sqlite_stmt_impl>();
    result.impl->db  = impl;
    result.impl->ptr = sqlite3_stmt_ptr{stmt_tmp};
    if(rc != SQLITE_OK)
        MIGRAPHX_THROW("error preparing '" + sql + "': " + impl->error_message());
    // sqlite succeeds without a statement for text that holds none, such as only a comment.
    assert(stmt_tmp != nullptr);
    return result;
}

void sqlite::set_busy_timeout(int ms) { sqlite3_busy_timeout(impl->get(), ms); }

bool sqlite::read_only() const { return sqlite3_db_readonly(impl->get(), "main") == 1; }

void sqlite_stmt::bind(int i, std::string_view s) const
{
    // A default-constructed string_view has null data(), and a null pointer binds SQL NULL
    // rather than an empty string, so empty input substitutes a valid pointer. SQLITE_TRANSIENT
    // makes sqlite take its own copy before returning, which is what lets callers bind
    // temporaries.
    const char* text = s.empty() ? "" : s.data();
    int rc = sqlite3_bind_text64(impl->get(), i, text, s.size(), SQLITE_TRANSIENT, SQLITE_UTF8);
    if(rc != SQLITE_OK)
        MIGRAPHX_THROW(impl->error_message());
}

void sqlite_stmt::bind(int i, std::int64_t x) const
{
    int rc = sqlite3_bind_int64(impl->get(), i, x);
    if(rc != SQLITE_OK)
        MIGRAPHX_THROW(impl->error_message());
}

void sqlite_stmt::bind(int i, const std::vector<char>& blob) const
{
    // As with text, an empty vector's data() may be null, which would bind SQL NULL; a
    // zero-length zeroblob is an empty BLOB instead. The 64-bit form is used because the
    // plain one takes the size as an int, and SQLITE_TRANSIENT copies before returning so
    // callers can bind temporaries.
    int rc = blob.empty()
                 ? sqlite3_bind_zeroblob(impl->get(), i, 0)
                 : sqlite3_bind_blob64(impl->get(), i, blob.data(), blob.size(), SQLITE_TRANSIENT);
    if(rc != SQLITE_OK)
        MIGRAPHX_THROW(impl->error_message());
}

std::size_t sqlite_stmt::parameter_count() const
{
    return sqlite3_bind_parameter_count(impl->get());
}

bool sqlite_stmt::step() const
{
    int rc = sqlite3_step(impl->get());
    if(rc == SQLITE_ROW)
        return true;
    if(rc == SQLITE_DONE)
        return false;
    MIGRAPHX_THROW(impl->error_message());
}

void sqlite_stmt::reset() const noexcept
{
    assert(impl != nullptr);
    // The return of sqlite3_reset is the error from the preceding step(), which the caller
    // has already seen as a throw. There is nothing new to report, and this must not throw.
    (void)sqlite3_reset(impl->get());
    (void)sqlite3_clear_bindings(impl->get());
}

/// Column i of the current row, keyed by its name. The values are built with parentheses
/// rather than the braces tidy suggests: value has an initializer_list constructor, which braces
/// would select, turning a keyed value into a two-element array.
static value column_value(sqlite3_stmt* stmt, int i)
{
    std::string name = sqlite3_column_name(stmt, i);
    auto type        = sqlite3_column_type(stmt, i);
    switch(type)
    {
    case SQLITE_INTEGER: return value(name, std::int64_t{sqlite3_column_int64(stmt, i)});
    // NOLINTNEXTLINE(modernize-return-braced-init-list)
    case SQLITE_FLOAT: return value(name, sqlite3_column_double(stmt, i));
    case SQLITE_TEXT:
    case SQLITE_BLOB: {
        // The data must be fetched before sqlite3_column_bytes: the other order can force a
        // conversion that invalidates the pointer. Text comes back unchanged through
        // sqlite3_column_blob, and a zero-length value as a null pointer, which means empty.
        const auto* data = static_cast<const char*>(sqlite3_column_blob(stmt, i));
        auto bytes       = sqlite3_column_bytes(stmt, i);
        assert(bytes >= 0);
        auto size = data == nullptr ? 0 : static_cast<std::size_t>(bytes);
        if(type == SQLITE_TEXT)
            // NOLINTNEXTLINE(modernize-return-braced-init-list)
            return value(name, size == 0 ? std::string{} : std::string(data, size));
        // NOLINTNEXTLINE(modernize-return-braced-init-list)
        return value(name, value::binary{data, size});
    }
    // NOLINTNEXTLINE(modernize-return-braced-init-list)
    default: return value(name, nullptr);
    }
}

value sqlite_stmt::to_value() const
{
    auto* stmt = impl->get();
    // Built as keyed values rather than a map, so a blob is moved into place instead of copied.
    std::vector<value> columns;
    auto indices = range(sqlite3_column_count(stmt));
    std::transform(indices.begin(),
                   indices.end(),
                   std::back_inserter(columns),
                   [&](std::ptrdiff_t i) { return column_value(stmt, static_cast<int>(i)); });
    // NOLINTNEXTLINE(modernize-return-braced-init-list)
    return value(columns, /* array_on_empty */ false);
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
