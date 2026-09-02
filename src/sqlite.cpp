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
#include <sqlite3.h>
#include <algorithm>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

using sqlite3_ptr = MIGRAPHX_MANAGE_PTR(sqlite3*, sqlite3_close);

struct sqlite_impl
{
    sqlite3* get() const { return ptr.get(); }

    // Returns false rather than throwing, so callers that treat an unusable database as
    // "no cache" do not have to catch. sqlite3_open_v2 hands back a handle even on failure
    // (that is where the error message lives), so `ptr` takes ownership either way.
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

    // Holding the connection keeps it alive for as long as any statement prepared on it,
    // since finalizing after the connection is closed is undefined.
    std::shared_ptr<sqlite_impl> db;
    sqlite3_stmt_ptr ptr;
};

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
    // Using '+' instead of bitwise '|' to avoid compilation warning
    r.impl->open(p, SQLITE_OPEN_READWRITE + SQLITE_OPEN_CREATE);
    return r;
}

optional<sqlite> sqlite::try_write(const fs::path& p)
{
    sqlite r;
    r.impl = std::make_shared<sqlite_impl>();
    // Using '+' instead of bitwise '|' to avoid compilation warning
    if(not r.impl->try_open(p, SQLITE_OPEN_READWRITE + SQLITE_OPEN_CREATE))
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

sqlite_stmt sqlite::prepare(const std::string& s)
{
    sqlite3_stmt* stmt_tmp = nullptr;
    int rc = sqlite3_prepare_v2(impl->get(), s.c_str(), -1, &stmt_tmp, nullptr);
    sqlite_stmt result;
    result.impl     = std::make_shared<sqlite_stmt_impl>();
    result.impl->db = impl;
    result.impl->ptr = sqlite3_stmt_ptr{stmt_tmp};
    if(rc != SQLITE_OK)
        MIGRAPHX_THROW("error preparing '" + s + "': " + impl->error_message());
    return result;
}

void sqlite::set_busy_timeout(int ms) { sqlite3_busy_timeout(impl->get(), ms); }

sqlite_stmt& sqlite_stmt::bind(int i, std::string_view s)
{
    // A null pointer binds SQL NULL rather than an empty string, and a default-constructed
    // string_view has null data(), so empty input needs its own case.
    int rc = s.empty() ? sqlite3_bind_text64(impl->get(), i, "", 0, SQLITE_STATIC, SQLITE_UTF8)
                       : sqlite3_bind_text64(
                             impl->get(), i, s.data(), s.size(), SQLITE_TRANSIENT, SQLITE_UTF8);
    if(rc != SQLITE_OK)
        MIGRAPHX_THROW(impl->error_message());
    return *this;
}

sqlite_stmt& sqlite_stmt::bind(int i, std::int64_t x)
{
    int rc = sqlite3_bind_int64(impl->get(), i, x);
    if(rc != SQLITE_OK)
        MIGRAPHX_THROW(impl->error_message());
    return *this;
}

sqlite_stmt& sqlite_stmt::bind(int i, const std::vector<char>& blob)
{
    // As with text, an empty vector's data() may be null, which would bind SQL NULL; a
    // zero-length zeroblob is an empty BLOB instead. bind_blob64 is used because the
    // non-64 form takes the size as an int.
    int rc = blob.empty()
                 ? sqlite3_bind_zeroblob(impl->get(), i, 0)
                 : sqlite3_bind_blob64(
                       impl->get(), i, blob.data(), blob.size(), SQLITE_TRANSIENT);
    if(rc != SQLITE_OK)
        MIGRAPHX_THROW(impl->error_message());
    return *this;
}

bool sqlite_stmt::step()
{
    int rc = sqlite3_step(impl->get());
    if(rc == SQLITE_ROW)
        return true;
    if(rc == SQLITE_DONE)
        return false;
    MIGRAPHX_THROW(impl->error_message());
}

void sqlite_stmt::reset() noexcept
{
    if(impl == nullptr)
        return;
    // The return of sqlite3_reset is the error from the preceding step(), which the caller
    // has already seen as a throw. There is nothing new to report, and this must not throw.
    (void)sqlite3_reset(impl->get());
    (void)sqlite3_clear_bindings(impl->get());
}

std::string sqlite_stmt::column_text(int i) const
{
    const auto* text = sqlite3_column_text(impl->get(), i);
    int n            = sqlite3_column_bytes(impl->get(), i);
    if(text == nullptr or n <= 0)
        return {};
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    return {reinterpret_cast<const char*>(text), static_cast<std::size_t>(n)};
}

std::vector<char> sqlite_stmt::column_blob(int i) const
{
    // sqlite3_column_blob must be called before sqlite3_column_bytes: the other order can
    // force a type conversion that invalidates the pointer.
    const auto* data = static_cast<const char*>(sqlite3_column_blob(impl->get(), i));
    int n            = sqlite3_column_bytes(impl->get(), i);
    if(data == nullptr or n <= 0)
        return {};
    return {data, data + n};
}

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
