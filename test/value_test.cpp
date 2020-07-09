#include <migraphx/value.hpp>
#include <test.hpp>

TEST_CASE(value_default_construct)
{
    migraphx::value v;
    EXPECT(v.is_null());
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_int1)
{
    EXPECT(migraphx::value(1).is_int64());
    migraphx::value v(1);
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 1);
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_int2)
{
    migraphx::value v = 1;
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 1);
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_string)
{
    migraphx::value v = "one";
    EXPECT(v.is_string());
    EXPECT(v.get_string() == "one");
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_assign_int)
{
    migraphx::value v;
    v = 0;
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 0);
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_copy_construct)
{
    migraphx::value v1(1);
    migraphx::value v2 = v1;
    EXPECT(v1 == v2);
}

TEST_CASE(value_copy_assign)
{
    migraphx::value v1(1);
    migraphx::value v2;
    v2 = v1;
    EXPECT(v1 == v2);
}

TEST_CASE(value_reassign)
{
    migraphx::value v1(1);
    migraphx::value v2 = v1;
    v1 = 2;
    EXPECT(v1 != v2);
}

TEST_CASE(value_construct_array)
{
    migraphx::value v = {1, 2, 3};
    EXPECT(v.is_array());
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front() == migraphx::value(1));
    EXPECT(v[1] == migraphx::value(2));
    EXPECT(v.at(1) == migraphx::value(2));
    EXPECT(v.back() == migraphx::value(3));
}

TEST_CASE(value_construct_key_int1)
{
    migraphx::value v("one", 1);
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 1);
    EXPECT(v.get_key() == "one");
}

TEST_CASE(value_construct_key_int2)
{
    migraphx::value v = {"one", 1};
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 1);
    EXPECT(v.get_key() == "one");
}

TEST_CASE(value_construct_object)
{
    migraphx::value v = {{"one", 1}, {"two", 2}, {"three", 3}};
    EXPECT(v.is_object());
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front().get_int64() == 1);
    EXPECT(v.front().get_key() == "one");
    EXPECT(v[1].is_int64());
    EXPECT(v[1].get_int64() == 2);
    EXPECT(v[1].get_key() == "two");
    EXPECT(v.back().is_int64());
    EXPECT(v.back().get_int64() == 3);
    EXPECT(v.back().get_key() == "three");

    EXPECT(v.contains("one"));
    EXPECT(v.contains("two"));
    EXPECT(v.contains("three"));
    EXPECT(not v.contains("four"));

    EXPECT(v.at("one").is_int64());
    EXPECT(v.at("one").get_int64() == 1);
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("two").is_int64());
    EXPECT(v.at("two").get_int64() == 2);
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("three").is_int64());
    EXPECT(v.at("three").get_int64() == 3);
    EXPECT(v.at("three").get_key() == "three");

    EXPECT(v["one"].is_int64());
    EXPECT(v["one"].get_int64() == 1);
    EXPECT(v["one"].get_key() == "one");
    EXPECT(v["two"].is_int64());
    EXPECT(v["two"].get_int64() == 2);
    EXPECT(v["two"].get_key() == "two");
    EXPECT(v["three"].is_int64());
    EXPECT(v["three"].get_int64() == 3);
    EXPECT(v["three"].get_key() == "three");
}

TEST_CASE(value_bracket_object)
{
    migraphx::value v;
    v["one"] = 1;
    v["two"] = 2;
    v["three"] = 3;

    EXPECT(v.is_object());
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front().get_int64() == 1);
    EXPECT(v.front().get_key() == "one");
    EXPECT(v[1].is_int64());
    EXPECT(v[1].get_int64() == 2);
    EXPECT(v[1].get_key() == "two");
    EXPECT(v.back().is_int64());
    EXPECT(v.back().get_int64() == 3);
    EXPECT(v.back().get_key() == "three");

    EXPECT(v.contains("one"));
    EXPECT(v.contains("two"));
    EXPECT(v.contains("three"));
    EXPECT(not v.contains("four"));

    EXPECT(v.at("one").is_int64());
    EXPECT(v.at("one").get_int64() == 1);
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("two").is_int64());
    EXPECT(v.at("two").get_int64() == 2);
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("three").is_int64());
    EXPECT(v.at("three").get_int64() == 3);
    EXPECT(v.at("three").get_key() == "three");
}

TEST_CASE(value_insert_object)
{
    migraphx::value v;
    v.insert({"one", 1});
    v.insert({"two", 2});
    v.insert({"three", 3});
    EXPECT(v.is_object());
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front().get_int64() == 1);
    EXPECT(v.front().get_key() == "one");
    EXPECT(v[1].is_int64());
    EXPECT(v[1].get_int64() == 2);
    EXPECT(v[1].get_key() == "two");
    EXPECT(v.back().is_int64());
    EXPECT(v.back().get_int64() == 3);
    EXPECT(v.back().get_key() == "three");

    EXPECT(v.contains("one"));
    EXPECT(v.contains("two"));
    EXPECT(v.contains("three"));
    EXPECT(not v.contains("four"));

    EXPECT(v.at("one").is_int64());
    EXPECT(v.at("one").get_int64() == 1);
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("two").is_int64());
    EXPECT(v.at("two").get_int64() == 2);
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("three").is_int64());
    EXPECT(v.at("three").get_int64() == 3);
    EXPECT(v.at("three").get_key() == "three");

    EXPECT(v["one"].is_int64());
    EXPECT(v["one"].get_int64() == 1);
    EXPECT(v["one"].get_key() == "one");
    EXPECT(v["two"].is_int64());
    EXPECT(v["two"].get_int64() == 2);
    EXPECT(v["two"].get_key() == "two");
    EXPECT(v["three"].is_int64());
    EXPECT(v["three"].get_int64() == 3);
    EXPECT(v["three"].get_key() == "three");
}

TEST_CASE(value_emplace_object)
{
    migraphx::value v;
    v.emplace("one", 1);
    v.emplace("two", 2);
    v.emplace("three", 3);
    EXPECT(v.is_object());
    EXPECT(v.size() == 3);
    EXPECT(not v.empty());
    EXPECT(v.data() != nullptr);
    EXPECT(v.front().is_int64());
    EXPECT(v.front().get_int64() == 1);
    EXPECT(v.front().get_key() == "one");
    EXPECT(v[1].is_int64());
    EXPECT(v[1].get_int64() == 2);
    EXPECT(v[1].get_key() == "two");
    EXPECT(v.back().is_int64());
    EXPECT(v.back().get_int64() == 3);
    EXPECT(v.back().get_key() == "three");

    EXPECT(v.contains("one"));
    EXPECT(v.contains("two"));
    EXPECT(v.contains("three"));
    EXPECT(not v.contains("four"));

    EXPECT(v.at("one").is_int64());
    EXPECT(v.at("one").get_int64() == 1);
    EXPECT(v.at("one").get_key() == "one");
    EXPECT(v.at("two").is_int64());
    EXPECT(v.at("two").get_int64() == 2);
    EXPECT(v.at("two").get_key() == "two");
    EXPECT(v.at("three").is_int64());
    EXPECT(v.at("three").get_int64() == 3);
    EXPECT(v.at("three").get_key() == "three");

    EXPECT(v["one"].is_int64());
    EXPECT(v["one"].get_int64() == 1);
    EXPECT(v["one"].get_key() == "one");
    EXPECT(v["two"].is_int64());
    EXPECT(v["two"].get_int64() == 2);
    EXPECT(v["two"].get_key() == "two");
    EXPECT(v["three"].is_int64());
    EXPECT(v["three"].get_int64() == 3);
    EXPECT(v["three"].get_key() == "three");
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
