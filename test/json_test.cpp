#include <migraphx/shape.hpp>
#include <migraphx/argument.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/serialize.hpp>
#include <migraphx/functional.hpp>
#include <migraphx/json.hpp>
#include <test.hpp>

TEST_CASE(empty_value)
{
    migraphx::value v;
    auto json_str = migraphx::to_json_string(v);
    EXPECT(json_str == "null");
}

TEST_CASE(empty_value_rev)
{
    std::string json_str = "null";
    migraphx::value v = migraphx::from_json_string(json_str);
    migraphx::value ev = migraphx::value();
    EXPECT(v == ev);
}


TEST_CASE(int_value)
{
    migraphx::value v    = -1;
    std::string json_str = migraphx::to_json_string(v);
    EXPECT(json_str == "-1");
}

TEST_CASE(int_value_rev)
{
    std::string json_str = "-1";
    migraphx::value v    = migraphx::from_json_string(json_str);
    migraphx::value ev   = -1;
    EXPECT(v == ev);
}

TEST_CASE(unsigned_value)
{
    migraphx::value v    = 1;
    std::string json_str = migraphx::to_json_string(v);
    EXPECT(json_str == "1");
}

TEST_CASE(unsigned_value_rev)
{
    std::string json_str = "1";
    migraphx::value v    = migraphx::from_json_string(json_str);
    EXPECT(v.is_uint64());
    EXPECT(v.get_uint64() == 1);
}

TEST_CASE(float_value)
{
    migraphx::value v    = 1.5;
    std::string json_str = migraphx::to_json_string(v);
    EXPECT(json_str == "1.5");
}

TEST_CASE(float_value_rev)
{
    std::string json_str = "1.5";
    migraphx::value v    = migraphx::from_json_string(json_str);
    migraphx::value ev   = 1.5;
    EXPECT(v == ev);
}

TEST_CASE(array_value)
{
    migraphx::value v    = {1, 2};
    std::string json_str = migraphx::to_json_string(v);
    EXPECT(json_str == "[1,2]");
}

TEST_CASE(array_value_rev)
{
    std::string json_str = "[1,2]";
    migraphx::value v    = migraphx::from_json_string(json_str);
    EXPECT(v.is_array());
    EXPECT(v.size() == 2);
    EXPECT(v[0].get_uint64() == 1);
    EXPECT(v[1].get_uint64() == 2);
}

TEST_CASE(object_value)
{
    migraphx::value v    = {{"a", 1.2}, {"b", true}};
    std::string json_str = migraphx::to_json_string(v);
    EXPECT(json_str == "{\"a\":1.2,\"b\":true}");
}

TEST_CASE(object_value_rev)
{
    std::string json_str = R"({"a":1.2,"b":true})";
    migraphx::value v    = migraphx::from_json_string(json_str);
    migraphx::value ev   = {{"a", 1.2}, {"b", true}};
    EXPECT(v == ev);
}

TEST_CASE(string_value)
{
    migraphx::value v    = "string_test";
    std::string json_str = migraphx::to_json_string(v);
    EXPECT(json_str == "\"string_test\"");
}

TEST_CASE(string_value_rev)
{
    std::string json_str = "\"string_test\"";
    migraphx::value v    = migraphx::from_json_string(json_str);
    migraphx::value ev   = "string_test";
    EXPECT(v == ev);
}

TEST_CASE(shape_value)
{
    migraphx::shape s{migraphx::shape::int32_type, {2, 3, 4, 5}};
    migraphx::value val     = migraphx::to_value(s);
    std::string json_str    = migraphx::to_json_string(val);
    migraphx::value val_rev = migraphx::from_json_string(json_str);
    migraphx::shape s_rev;
    migraphx::from_value(val_rev, s_rev);

    EXPECT(s == s_rev);
}

TEST_CASE(argument_value)
{
    migraphx::shape s{migraphx::shape::int32_type, {2, 3, 4, 5}};
    std::vector<int> data(s.elements());
    std::iota(data.begin(), data.end(), 1);
    migraphx::argument argu = migraphx::argument(s, data.data());

    migraphx::value val     = migraphx::to_value(argu);
    std::string json_str    = migraphx::to_json_string(val);
    migraphx::value val_rev = migraphx::from_json_string(json_str);
    migraphx::argument argu_rev;
    migraphx::from_value(val_rev, argu_rev);

    EXPECT(argu == argu_rev);
}

TEST_CASE(literal_value)
{
    migraphx::shape s{migraphx::shape::int32_type, {2, 3, 4, 5}};
    std::vector<int> data(s.elements());
    std::iota(data.begin(), data.end(), 1);
    migraphx::literal l = migraphx::literal(s, data);

    migraphx::value val     = migraphx::to_value(l);
    std::string json_str    = migraphx::to_json_string(val);
    migraphx::value val_rev = migraphx::from_json_string(json_str);
    migraphx::literal l_rev;
    migraphx::from_value(val_rev, l_rev);

    EXPECT(l == l_rev);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
