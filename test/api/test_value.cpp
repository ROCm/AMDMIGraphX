#include <cstdint>
#include <migraphx/migraphx.h>
#include <migraphx/migraphx.hpp>
#include "test.hpp"

TEST_CASE(value_default_construct)
{
    migraphx::value v;
    EXPECT(v.is_null());
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_int1)
{
    EXPECT(migraphx::value(int64_t{1}).is_int64());
    migraphx::value v(int64_t{1});
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 1);
    EXPECT(*v.if_int64() == 1);
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_int2)
{
    migraphx::value v = int64_t{1};
    EXPECT(v.is_int64());
    EXPECT(v.get_int64() == 1);
    EXPECT(*v.if_int64() == 1);
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_string)
{
    migraphx::value v = std::string{"one"};
    EXPECT(v.is_string());
    EXPECT(v.get_string() == "one");
    EXPECT(v.if_string() == "one");
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_key_string_literal_pair)
{
    // Use parens instead {} to construct to test the key-pair constructor
    migraphx::value v("key", std::string{"one"});
    EXPECT(v.is_string());
    EXPECT(v.get_string() == "one");
    EXPECT(v.if_string() == "one");
    EXPECT(v.get_key() == "key");
}

TEST_CASE(value_construct_float)
{
    migraphx::value v = 1.0;
    EXPECT(v.is_float());
    // TODO: add float_equal method
    EXPECT(v.get_key().empty());
}

TEST_CASE(value_construct_bool)
{
    migraphx::value v = true;
    EXPECT(v.is_bool());
    EXPECT(v.get_bool() == true);
    EXPECT(v.get_key().empty());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
