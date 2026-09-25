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
#include <migraphx/errors.hpp>
#include <migraphx/filesystem.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/iterator.hpp>
#include <migraphx/optional.hpp>
#include <migraphx/value.hpp>
#include <cassert>
#include <cstdint>
#include <iterator>
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
/// never outlive it. Copies share the same statement.
///
/// Calling it with arguments runs it: the arguments are bound to the parameters in order and the
/// result comes back as a range of rows. Since copies share one statement, only the rows of one
/// call may be alive at a time.
///
/// Not thread safe: use a statement from one thread at a time.
struct MIGRAPHX_EXPORT sqlite_stmt
{
    struct rows;

    sqlite_stmt() = default;

    /// Run the statement with xs bound to its parameters in order, and return its rows. Defined
    /// after rows, which has to be complete for a function returning it to be defined.
    template <class... Ts>
    rows operator()(const Ts&... xs) const;

    bool valid() const { return impl != nullptr; }

    private:
    // Parameter indices are 1-based, matching sqlite's own convention.
    void bind(int i, std::string_view s) const;
    void bind(int i, std::int64_t x) const;
    void bind(int i, const std::vector<char>& blob) const;

    std::size_t parameter_count() const;

    /// Step once. True when a row is available, false when the statement is done.
    bool step() const;

    /// Clear bindings and rewind, so the statement can be used again. Safe at any point,
    /// including after step() has thrown.
    void reset() const noexcept;

    /// The current row as an object keyed by column name. Blobs become value::binary and SQL
    /// NULL becomes a null value.
    value to_value() const;

    friend struct sqlite;
    std::shared_ptr<sqlite_stmt_impl> impl;
};

/// The rows produced by one call of a statement, as an input range of values.
///
/// The first row is fetched when the call is made, so a statement that returns nothing, such as
/// an insert, has already run by the time the call returns, whether or not the range is
/// iterated. The statement is reset when the range is destroyed: an unfinished select holds a
/// read lock on the database until then, which would stall writers in other processes.
struct sqlite_stmt::rows
{
    // Only ever a prvalue returned from a call, so it never needs copying or moving, and a copy
    // would reset the statement out from under the original.
    rows(const rows&)            = delete;
    rows(rows&&)                 = delete;
    rows& operator=(const rows&) = delete;
    rows& operator=(rows&&)      = delete;
    ~rows() { stmt.reset(); }

    struct iterator : iterator_operators<iterator>
    {
        using value_type        = value;
        using reference         = value_type;
        using difference_type   = std::ptrdiff_t;
        using iterator_category = std::input_iterator_tag;
        using pointer           = value*;

        iterator() = default;

        iterator(const rows* pparent, bool pavailable) : parent(pparent), available(pavailable) {}

        reference operator*() const
        {
            assert(parent != nullptr and available);
            return parent->stmt.to_value();
        }

        template <class U>
        static void increment(U& x)
        {
            assert(x.parent != nullptr and x.available);
            x.available = x.parent->stmt.step();
        }

        template <class U, class V>
        static auto equal(const U& x, const V& y)
        {
            return x.parent == y.parent and x.available == y.available;
        }

        private:
        const rows* parent = nullptr;
        bool available     = false;
    };

    iterator begin() const { return {this, first}; }
    iterator end() const { return {this, false}; }

    private:
    friend struct sqlite_stmt;
    explicit rows(sqlite_stmt s) : stmt(std::move(s)), first(stmt.step()) {}

    sqlite_stmt stmt;
    bool first = false;
};

template <class... Ts>
sqlite_stmt::rows sqlite_stmt::operator()(const Ts&... xs) const
{
    if(not valid())
        MIGRAPHX_THROW("sqlite: calling a statement that was never prepared");
    assert(sizeof...(Ts) == parameter_count());
    // Anything left from the previous call, bindings or an unfinished result, goes first.
    reset();
    int i = 0;
    each_args([&](const auto& x) { bind(++i, x); }, xs...);
    return rows{*this};
}

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

    /// True when writes will be refused. Opening for writing still succeeds on a file the OS
    /// has write-protected, in which case sqlite quietly opens it read-only; this is how to tell.
    bool read_only() const;

    private:
    std::shared_ptr<sqlite_impl> impl;
};

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
#endif // MIGRAPHX_GUARD_MIGRAPHX_SQLITE_HPP
