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
#include <migraphx/tmp_dir.hpp>
#include <test.hpp>
#include <cstdint>
#include <string_view>
#include <vector>

TEST_CASE(read_write)
{
    const std::string create_table = R"__migraphx__(
    CREATE TABLE IF NOT EXISTS test_db (
    id INTEGER PRIMARY KEY ASC,
    data TEXT NOT NULL
    );
    INSERT INTO test_db (id, data) VALUES (1, "a");
    )__migraphx__";

    const std::string select_all = R"__migraphx__(
    SELECT * FROM test_db;
    )__migraphx__";

    migraphx::tmp_dir td{};
    auto db_path = td.path / "test.db";
    {
        auto db = migraphx::sqlite::write(db_path);
        db.execute(create_table);
    }
    {
        auto db   = migraphx::sqlite::read(db_path);
        auto rows = db.execute(select_all);
        EXPECT(rows.size() == 1);
        const auto& row = rows.front();
        EXPECT(row.at("data") == "a");
        EXPECT(row.at("id") == "1");
    }
}

TEST_CASE(prepared_blob_round_trip)
{
    // Bytes that raw SQL text cannot carry: an embedded NUL and a single quote. This is the
    // reason binaries need parameter binding rather than string interpolation.
    const std::vector<char> blob{'\0', 'a', '\'', '\0', static_cast<char>(0xff), 'z'};

    migraphx::tmp_dir td{};
    auto db_path = td.path / "blob.db";
    {
        auto db = migraphx::sqlite::write(db_path);
        db.execute(R"__migraphx__(
        CREATE TABLE IF NOT EXISTS blob_db (
        name TEXT PRIMARY KEY,
        size INTEGER NOT NULL,
        data BLOB NOT NULL
        );
        )__migraphx__");

        // One statement, two inserts: the reset/rebind path backends rely on.
        auto insert = db.prepare("INSERT INTO blob_db (name, size, data) VALUES (?, ?, ?);");
        EXPECT(insert.valid());

        insert.bind(1, std::string_view{"k1"})
            .bind(2, static_cast<std::int64_t>(blob.size()))
            .bind(3, blob);
        EXPECT(not insert.step());
        insert.reset();

        insert.bind(1, std::string_view{"empty"})
            .bind(2, static_cast<std::int64_t>(0))
            .bind(3, std::vector<char>{});
        EXPECT(not insert.step());
        insert.reset();
    }
    {
        auto db     = migraphx::sqlite::read(db_path);
        auto select = db.prepare("SELECT name, data FROM blob_db WHERE name = ?;");

        {
            migraphx::sqlite_stmt_reset guard{select};
            select.bind(1, std::string_view{"k1"});
            EXPECT(select.step());
            EXPECT(select.column_text(0) == "k1");
            EXPECT((select.column_blob(1) == blob));
            EXPECT(not select.step());
        }
        {
            // An empty blob must come back as an empty blob, not as NULL.
            migraphx::sqlite_stmt_reset guard{select};
            select.bind(1, std::string_view{"empty"});
            EXPECT(select.step());
            EXPECT(select.column_blob(1).empty());
        }
        {
            migraphx::sqlite_stmt_reset guard{select};
            select.bind(1, std::string_view{"missing"});
            EXPECT(not select.step());
        }
    }
}

TEST_CASE(try_write_unusable_path)
{
    migraphx::tmp_dir td{};
    // A directory component that is really a file, so the database can never be created.
    auto blocker = td.path / "not_a_dir";
    {
        auto db = migraphx::sqlite::write(blocker);
        db.execute("CREATE TABLE IF NOT EXISTS t (id INTEGER PRIMARY KEY ASC);");
    }
    EXPECT(not migraphx::sqlite::try_write(blocker / "nested.db").has_value());
    EXPECT(migraphx::sqlite::try_write(td.path / "ok.db").has_value());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
