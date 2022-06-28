#include <migraphx/sqlite.hpp>
#include <migraphx/tmp_dir.hpp>
#include <test.hpp>

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

TEST_CASE(read_write)
{
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
        auto row = rows.front();
        EXPECT(row.at("data") == "a");
        EXPECT(row.at("id") == "1");
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
