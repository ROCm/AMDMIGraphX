#include <migraphx/msgpack.hpp>
#include <migraphx/value.hpp>
#include <msgpack.hpp>
#include <map>
#include "test.hpp"

template <class T>
std::vector<char> msgpack_buffer(const T& src)
{
    std::stringstream buffer;
    msgpack::pack(buffer, src);
    buffer.seekg(0);
    std::string str = buffer.str();
    return std::vector<char>(str.data(), str.data() + str.size());
}

TEST_CASE(test_msgpack_empty_value)
{
    migraphx::value v;
    auto buffer = migraphx::to_msgpack(v);
    auto mp     = migraphx::from_msgpack(buffer);
    EXPECT(mp == v);
    EXPECT(v.is_null());
    EXPECT(mp.is_null());
}

TEST_CASE(test_msgpack_int)
{
    migraphx::value v = 3;
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(3));
    EXPECT(migraphx::from_msgpack(buffer).to<int>() == v.to<int>());
}

TEST_CASE(test_msgpack_int_negative)
{
    migraphx::value v = -3;
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(-3));
    EXPECT(migraphx::from_msgpack(buffer).to<int>() == v.to<int>());
}

TEST_CASE(test_msgpack_float)
{
    migraphx::value v = 3.0;
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(3.0));
    EXPECT(migraphx::from_msgpack(buffer) == v);
}

TEST_CASE(test_msgpack_string)
{
    migraphx::value v = "abc";
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer("abc"));
    EXPECT(migraphx::from_msgpack(buffer) == v);
}

TEST_CASE(test_msgpack_array)
{
    migraphx::value v = {1, 2, 3};
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(std::vector<int>{1, 2, 3}));
    EXPECT(migraphx::from_msgpack(buffer).to_vector<int>() == v.to_vector<int>());
}

TEST_CASE(test_msgpack_object)
{
    migraphx::value v = {{"one", 1.0}, {"three", 3.0}, {"two", 2.0}};
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(std::map<std::string, double>{
                         {"one", 1.0}, {"three", 3.0}, {"two", 2.0}}));
    EXPECT(migraphx::from_msgpack(buffer) == v);
}

struct foo
{
    double a;
    std::string b;
    MSGPACK_DEFINE_MAP(a, b);
};

TEST_CASE(test_msgpack_object_class)
{
    migraphx::value v = {{"a", 1.0}, {"b", "abc"}};
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(foo{1.0, "abc"}));
    EXPECT(migraphx::from_msgpack(buffer) == v);
}

TEST_CASE(test_msgpack_array_class)
{
    migraphx::value v = {{{"a", 1.0}, {"b", "abc"}}, {{"a", 3.0}, {"b", "xyz"}}};
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(std::vector<foo>{foo{1.0, "abc"}, foo{3.0, "xyz"}}));
    EXPECT(migraphx::from_msgpack(buffer) == v);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
