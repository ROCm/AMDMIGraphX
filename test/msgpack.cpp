/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2022 Advanced Micro Devices, Inc. All rights reserved.
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
    return std::vector<char>(str.data(), str.data() + str.size()); // NOLINT
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

TEST_CASE(test_msgpack_bool)
{
    migraphx::value v = true;
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(true));
    EXPECT(migraphx::from_msgpack(buffer) == v);
}

TEST_CASE(test_msgpack_float)
{
    migraphx::value v = 3.0;
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(3.0));
    double epsilon = 1e-9;
    EXPECT(std::abs(migraphx::from_msgpack(buffer).to<float>() - v.to<float>()) < epsilon);
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

TEST_CASE(test_msgpack_empty_array)
{
    migraphx::value v = migraphx::value::array{};
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(std::vector<int>{}));
    EXPECT(migraphx::from_msgpack(buffer) == v);
}

TEST_CASE(test_msgpack_object)
{
    migraphx::value v = {{"one", 1.0}, {"three", 3.0}, {"two", 2.0}};
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(std::map<std::string, double>{
                         {"one", 1.0}, {"three", 3.0}, {"two", 2.0}}));

    // converted to vector in the following line because in value.cpp value constructor with
    // unordered map is creating vector<value> with map items as vector elements
    // value(std::vector<value>(m.begin(), m.end()), false)
    EXPECT(migraphx::from_msgpack(buffer).to_vector<double>() == v.to_vector<double>());
}

TEST_CASE(test_msgpack_empty_object)
{
    migraphx::value v = migraphx::value::object{};
    auto buffer       = migraphx::to_msgpack(v);
    EXPECT(buffer == msgpack_buffer(std::vector<int>{}));
    auto u = migraphx::from_msgpack(buffer);
    // This is not equal since an empty object becomes an empty array
    EXPECT(u != v);
    EXPECT(u.is_array());
    EXPECT(u.size() == 0);
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
