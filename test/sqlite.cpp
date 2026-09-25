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
#include <vector>

/// Every row a call produced, so a test can count and inspect them.
static std::vector<migraphx::value> collect(const migraphx::sqlite_stmt::rows& r)
{
    return std::vector<migraphx::value>(r.begin(), r.end());
}

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

        // One statement, two inserts: calling it again rebinds, which is what backends rely on.
        // An insert produces no rows, and runs whether or not they are iterated.
        auto insert = db.prepare("INSERT INTO blob_db (name, size, data) VALUES (?, ?, ?);");
        EXPECT(insert.valid());
        EXPECT(collect(insert("k1", static_cast<std::int64_t>(blob.size()), blob)).empty());
        insert("empty", std::int64_t{0}, std::vector<char>{});
    }
    {
        auto db     = migraphx::sqlite::read(db_path);
        auto select = db.prepare("SELECT name, size, data FROM blob_db WHERE name = ?;");

        auto found = collect(select("k1"));
        EXPECT(found.size() == 1);
        EXPECT(found.front().at("name").get_string() == "k1");
        EXPECT(found.front().at("size").to<std::size_t>() == blob.size());
        EXPECT(found.front().at("data").get_binary() == migraphx::value::binary{blob});

        // An empty blob must come back as an empty blob, not as NULL.
        auto empty = collect(select("empty"));
        EXPECT(empty.size() == 1);
        EXPECT(empty.front().at("data").is_binary());
        EXPECT(empty.front().at("data").get_binary().empty());

        EXPECT(collect(select("missing")).empty());
    }
}

// A select abandoned after its first row must not keep holding the database. Until the
// statement is reset it holds a read lock, and a writer on another connection would wait out
// its busy timeout and then fail.
TEST_CASE(abandoned_rows_release_the_database)
{
    migraphx::tmp_dir td{};
    auto db_path = td.path / "lock.db";
    auto writer  = migraphx::sqlite::write(db_path);
    writer.execute("CREATE TABLE t (id INTEGER PRIMARY KEY);"
                   "INSERT INTO t (id) VALUES (1), (2);");

    auto reader = migraphx::sqlite::read(db_path);
    auto select = reader.prepare("SELECT id FROM t;");
    {
        // Read one of the two rows and stop.
        auto rows = select();
        EXPECT(rows.begin() != rows.end());
    }

    auto insert = writer.prepare("INSERT INTO t (id) VALUES (?);");
    insert(std::int64_t{3});
    EXPECT(writer.execute("SELECT id FROM t;").size() == 3);
}

TEST_CASE(unprepared_statement_throws)
{
    migraphx::sqlite_stmt stmt;
    EXPECT(not stmt.valid());
    EXPECT(test::throws([&] { stmt(); }));
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
