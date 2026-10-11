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
#include <migraphx/small_vector.hpp>
#include <algorithm>
#include <array>
#include <functional>
#include <iterator>
#include <list>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <test.hpp>

// True when the elements live inside the vector object itself
template <class V>
static bool is_inline(const V& v)
{
    const void* first = &v;
    const void* last  = std::next(&v);
    const void* p     = v.data();
    return std::less_equal<const void*>{}(first, p) and std::less<const void*>{}(p, last);
}

// Counts live instances to check that every constructed element is destroyed exactly once
struct counted
{
    static inline int live = 0;
    int value;
    counted() : counted(0) {}
    counted(int v) : value(v) { ++live; }
    counted(const counted& other) : value(other.value) { ++live; }
    counted(counted&& other) noexcept : value(other.value) { ++live; }
    counted& operator=(const counted&) = default;
    counted& operator=(counted&&)      = default;
    ~counted() { --live; }
    friend bool operator==(const counted& x, const counted& y) { return x.value == y.value; }
    friend std::ostream& operator<<(std::ostream& os, const counted& x) { return os << x.value; }
};

// Copy-only type whose copy throws once the budget runs out, to check the strong guarantee
struct throwing_copy
{
    static inline int budget = 0;
    int value;
    throwing_copy(int v) : value(v) {}
    throwing_copy(const throwing_copy& other) : value(other.value)
    {
        if(budget == 0)
            throw std::runtime_error("copy budget exhausted");
        --budget;
    }
    throwing_copy& operator=(const throwing_copy&) = default;
    friend bool operator==(const throwing_copy& x, const throwing_copy& y)
    {
        return x.value == y.value;
    }
    friend std::ostream& operator<<(std::ostream& os, const throwing_copy& x)
    {
        return os << x.value;
    }
};

static std::vector<int> iota_vector(int n)
{
    std::vector<int> v(n);
    std::iota(v.begin(), v.end(), 0);
    return v;
}

TEST_CASE(default_size)
{
    static_assert(migraphx::small_vector_default_size<char>() == 32);
    static_assert(migraphx::small_vector_default_size<int>() == 8);
    static_assert(migraphx::small_vector_default_size<double>() == 4);
    static_assert(migraphx::small_vector_default_size<std::array<char, 16>>() == 2);
    static_assert(migraphx::small_vector_default_size<std::array<char, 24>>() == 2);
    static_assert(migraphx::small_vector_default_size<std::array<char, 64>>() == 2);
    static_assert(std::is_same<migraphx::small_vector<int>, migraphx::small_vector<int, 8>>{});
    static_assert(
        std::is_same<migraphx::small_vector<double>, migraphx::small_vector<double, 4>>{});
}

TEST_CASE(default_construct)
{
    migraphx::small_vector<int> v;
    EXPECT(v.empty());
    EXPECT(v.size() == 0);
    EXPECT(v.capacity() == 8);
    EXPECT(v.begin() == v.end());
    EXPECT(v.data() != nullptr);
    EXPECT(is_inline(v));
}

TEST_CASE(construct_count)
{
    migraphx::small_vector<int, 4> v(3);
    EXPECT(v.size() == 3);
    EXPECT(v == migraphx::small_vector<int, 4>{0, 0, 0});
    EXPECT(is_inline(v));

    migraphx::small_vector<int, 4> w(6);
    EXPECT(w.size() == 6);
    EXPECT(std::all_of(w.begin(), w.end(), [](int x) { return x == 0; }));
    EXPECT(not is_inline(w));
}

TEST_CASE(construct_count_value)
{
    migraphx::small_vector<int, 4> v(3, 7);
    EXPECT(v == migraphx::small_vector<int, 4>{7, 7, 7});
    EXPECT(is_inline(v));

    migraphx::small_vector<int, 4> w(6, 7);
    EXPECT(w == migraphx::small_vector<int, 4>{7, 7, 7, 7, 7, 7});
    EXPECT(w.capacity() == 6);
    EXPECT(not is_inline(w));
}

TEST_CASE(construct_range_forward)
{
    auto data = iota_vector(6);
    migraphx::small_vector<int, 4> v(data.begin(), data.end());
    EXPECT(v == migraphx::small_vector<int, 4>{0, 1, 2, 3, 4, 5});
    EXPECT(v.capacity() == 6);

    std::list<int> l{1, 2, 3};
    migraphx::small_vector<int, 4> w(l.begin(), l.end());
    EXPECT(w == migraphx::small_vector<int, 4>{1, 2, 3});
    EXPECT(is_inline(w));
}

TEST_CASE(construct_range_input)
{
    std::istringstream ss("1 2 3 4 5");
    migraphx::small_vector<int, 2> v(std::istream_iterator<int>{ss}, std::istream_iterator<int>{});
    EXPECT(v == migraphx::small_vector<int, 2>{1, 2, 3, 4, 5});
}

TEST_CASE(construct_initializer_list)
{
    migraphx::small_vector<int, 4> v{1, 2, 3};
    EXPECT(v.size() == 3);
    EXPECT(v[0] == 1);
    EXPECT(v[1] == 2);
    EXPECT(v[2] == 3);
    EXPECT(is_inline(v));

    migraphx::small_vector<int, 4> w{1, 2, 3, 4, 5};
    EXPECT(w.size() == 5);
    EXPECT(not is_inline(w));

    // Braces pick the initializer list over the count constructor
    migraphx::small_vector<int, 4> x{3};
    EXPECT(x == migraphx::small_vector<int, 4>{3});
}

TEST_CASE(construct_deduction)
{
    auto data = iota_vector(3);
    migraphx::small_vector v(data.begin(), data.end());
    static_assert(std::is_same<decltype(v), migraphx::small_vector<int>>{});
    EXPECT(v == migraphx::small_vector<int>{0, 1, 2});

    migraphx::small_vector w{1.0, 2.0};
    static_assert(std::is_same<decltype(w), migraphx::small_vector<double>>{});
}

TEST_CASE(copy_construct_inline)
{
    migraphx::small_vector<int, 4> v{1, 2, 3};
    migraphx::small_vector<int, 4> w(v);
    EXPECT(w == v);
    EXPECT(is_inline(w));
    EXPECT(w.data() != v.data());
    w[0] = 9;
    EXPECT(v[0] == 1);
}

TEST_CASE(copy_construct_heap)
{
    migraphx::small_vector<int, 2> v{1, 2, 3, 4, 5, 6};
    v.reserve(32);
    migraphx::small_vector<int, 2> w(v);
    EXPECT(w == v);
    EXPECT(not is_inline(w));
    EXPECT(w.capacity() == 6);
    EXPECT(w.data() != v.data());
}

TEST_CASE(move_construct_inline)
{
    migraphx::small_vector<std::string, 4> v{"a", "b", "c"};
    migraphx::small_vector<std::string, 4> w(std::move(v));
    EXPECT(w == migraphx::small_vector<std::string, 4>{"a", "b", "c"});
    EXPECT(is_inline(w));
    EXPECT(v.empty()); // cppcheck-suppress accessMoved
}

TEST_CASE(move_construct_heap)
{
    migraphx::small_vector<int, 2> v{1, 2, 3, 4};
    const int* data = v.data();
    migraphx::small_vector<int, 2> w(std::move(v));
    EXPECT(w == migraphx::small_vector<int, 2>{1, 2, 3, 4});
    EXPECT(w.data() == data);
    EXPECT(v.empty()); // cppcheck-suppress accessMoved
    EXPECT(v.capacity() == 2);
    EXPECT(is_inline(v));
    // The moved-from vector is usable again
    v.push_back(5); // NOLINT(bugprone-use-after-move)
    EXPECT(v == migraphx::small_vector<int, 2>{5});
}

TEST_CASE(copy_assign)
{
    migraphx::small_vector<int, 4> v{1, 2, 3};
    migraphx::small_vector<int, 4> w{9};
    w = v;
    EXPECT(w == v);
    EXPECT(is_inline(w));

    migraphx::small_vector<int, 4> x{1, 2, 3, 4, 5, 6};
    w = x;
    EXPECT(w == x);
    EXPECT(not is_inline(w));

    // Assigning fewer elements reuses the heap buffer
    const int* data = w.data();
    w               = v;
    EXPECT(w == v);
    EXPECT(w.data() == data);
}

TEST_CASE(move_assign_inline)
{
    migraphx::small_vector<std::string, 4> v{"a", "b"};
    migraphx::small_vector<std::string, 4> w{"x", "y", "z"};
    w = std::move(v);
    EXPECT(w == migraphx::small_vector<std::string, 4>{"a", "b"});
    EXPECT(v.empty()); // cppcheck-suppress accessMoved
}

TEST_CASE(move_assign_heap)
{
    migraphx::small_vector<int, 2> v{1, 2, 3, 4};
    const int* data = v.data();
    migraphx::small_vector<int, 2> w{7, 8, 9};
    w = std::move(v);
    EXPECT(w == migraphx::small_vector<int, 2>{1, 2, 3, 4});
    EXPECT(w.data() == data);
    EXPECT(v.empty()); // cppcheck-suppress accessMoved
    EXPECT(is_inline(v));
}

TEST_CASE(assign_initializer_list)
{
    migraphx::small_vector<int, 4> v{9, 9};
    v = {1, 2, 3};
    EXPECT(v == migraphx::small_vector<int, 4>{1, 2, 3});
    v.assign({4, 5});
    EXPECT(v == migraphx::small_vector<int, 4>{4, 5});
}

TEST_CASE(assign_count)
{
    migraphx::small_vector<int, 4> v{1, 2, 3};
    v.assign(2, 7);
    EXPECT(v == migraphx::small_vector<int, 4>{7, 7});
    v.assign(6, 8);
    EXPECT(v == migraphx::small_vector<int, 4>{8, 8, 8, 8, 8, 8});
    EXPECT(not is_inline(v));
    v.assign(0, 1);
    EXPECT(v.empty());
}

TEST_CASE(assign_range)
{
    migraphx::small_vector<int, 4> v{1, 2, 3};
    auto data = iota_vector(5);
    v.assign(data.begin(), data.end());
    EXPECT(v == migraphx::small_vector<int, 4>{0, 1, 2, 3, 4});

    std::list<int> l{5, 6};
    v.assign(l.begin(), l.end());
    EXPECT(v == migraphx::small_vector<int, 4>{5, 6});

    std::istringstream ss("7 8 9");
    v.assign(std::istream_iterator<int>{ss}, std::istream_iterator<int>{});
    EXPECT(v == migraphx::small_vector<int, 4>{7, 8, 9});
}

TEST_CASE(self_assign)
{
    migraphx::small_vector<int, 2> v{1, 2, 3};
    const auto& alias = v;
    v                 = alias;
    EXPECT(v == migraphx::small_vector<int, 2>{1, 2, 3});
}

TEST_CASE(element_access)
{
    migraphx::small_vector<int, 4> v{1, 2, 3};
    const auto& cv = v;
    EXPECT(v.at(0) == 1);
    EXPECT(cv.at(2) == 3);
    EXPECT(v[1] == 2);
    EXPECT(cv[1] == 2);
    EXPECT(v.front() == 1);
    EXPECT(cv.front() == 1);
    EXPECT(v.back() == 3);
    EXPECT(cv.back() == 3);
    EXPECT(v.data() == &v[0]);
    EXPECT(cv.data() == &cv[0]);
    v.at(1)   = 5;
    v.front() = 4;
    v.back()  = 6;
    EXPECT(v == migraphx::small_vector<int, 4>{4, 5, 6});
    EXPECT(test::throws<std::out_of_range>([&] { v.at(3); }));
    EXPECT(test::throws<std::out_of_range>([&] { cv.at(3); }));
}

TEST_CASE(iterators)
{
    migraphx::small_vector<int, 4> v{1, 2, 3};
    const auto& cv = v;
    EXPECT(std::distance(v.begin(), v.end()) == 3);
    EXPECT(std::distance(cv.begin(), cv.end()) == 3);
    EXPECT(std::distance(v.cbegin(), v.cend()) == 3);
    EXPECT(*v.begin() == 1);
    EXPECT(*std::prev(v.end()) == 3);
    EXPECT(*v.rbegin() == 3);
    EXPECT(*cv.rbegin() == 3);
    EXPECT(*std::prev(v.rend()) == 1);
    EXPECT(*v.crbegin() == 3);
    EXPECT(*std::prev(v.crend()) == 1);
    std::vector<int> reversed(v.rbegin(), v.rend());
    EXPECT(reversed == std::vector<int>{3, 2, 1});
    EXPECT(std::accumulate(v.begin(), v.end(), 0) == 6);
}

TEST_CASE(reserve)
{
    migraphx::small_vector<int, 4> v{1, 2};
    v.reserve(3);
    EXPECT(v.capacity() == 4);
    EXPECT(is_inline(v));
    v.reserve(10);
    EXPECT(v.capacity() == 10);
    EXPECT(not is_inline(v));
    EXPECT(v == migraphx::small_vector<int, 4>{1, 2});
    const int* data = v.data();
    v.reserve(5);
    EXPECT(v.capacity() == 10);
    EXPECT(v.data() == data);
    EXPECT(test::throws<std::length_error>([&] { v.reserve(v.max_size() + 1); }));
}

TEST_CASE(shrink_to_fit)
{
    migraphx::small_vector<int, 4> v{1, 2, 3};
    v.shrink_to_fit();
    EXPECT(is_inline(v));

    v.reserve(20);
    EXPECT(not is_inline(v));
    v.shrink_to_fit();
    EXPECT(is_inline(v));
    EXPECT(v.capacity() == 4);
    EXPECT(v == migraphx::small_vector<int, 4>{1, 2, 3});

    v.assign({1, 2, 3, 4, 5, 6});
    v.reserve(20);
    v.shrink_to_fit();
    EXPECT(not is_inline(v));
    EXPECT(v.capacity() == 6);
    EXPECT(v == migraphx::small_vector<int, 4>{1, 2, 3, 4, 5, 6});

    v.clear();
    v.shrink_to_fit();
    EXPECT(is_inline(v));
    EXPECT(v.capacity() == 4);
}

TEST_CASE(max_size)
{
    migraphx::small_vector<int> v;
    EXPECT(v.max_size() ==
           std::allocator_traits<std::allocator<int>>::max_size(std::allocator<int>{}));
    EXPECT(v.max_size() == migraphx::small_vector<int, 2>{}.max_size());
}

TEST_CASE(clear_keeps_capacity)
{
    migraphx::small_vector<int, 2> v{1, 2, 3, 4};
    auto cap = v.capacity();
    v.clear();
    EXPECT(v.empty());
    EXPECT(v.capacity() == cap);
    EXPECT(not is_inline(v));
}

TEST_CASE(push_back_grow)
{
    migraphx::small_vector<int, 4> v;
    std::vector<std::size_t> caps;
    for(int i = 0; i < 20; i++)
    {
        v.push_back(i);
        caps.push_back(v.capacity());
    }
    EXPECT(v.size() == 20);
    EXPECT(std::equal(v.begin(), v.end(), iota_vector(20).begin()));
    EXPECT(std::is_sorted(caps.begin(), caps.end()));
    EXPECT(caps.front() == 4);
    EXPECT(caps.back() >= 20);
    EXPECT(not is_inline(v));
}

TEST_CASE(push_back_aliasing)
{
    migraphx::small_vector<std::string, 2> v{"first", "second"};
    v.push_back(v.front());
    EXPECT(v == migraphx::small_vector<std::string, 2>{"first", "second", "first"});
    v.push_back(v.back());
    EXPECT(v == migraphx::small_vector<std::string, 2>{"first", "second", "first", "first"});
    v.push_back(std::move(v[1]));
    EXPECT(v.size() == 5);
    EXPECT(v[4] == "second");
}

TEST_CASE(emplace_back)
{
    migraphx::small_vector<std::string, 2> v;
    auto& r1 = v.emplace_back(3, 'a');
    EXPECT(r1 == "aaa");
    EXPECT(&r1 == &v.back());
    auto& r2 = v.emplace_back("b");
    EXPECT(r2 == "b");
    auto& r3 = v.emplace_back();
    EXPECT(r3.empty());
    EXPECT(&r3 == &v.back());
    EXPECT(v.size() == 3);
}

TEST_CASE(pop_back)
{
    migraphx::small_vector<int, 4> v{1, 2, 3};
    v.pop_back();
    EXPECT(v == migraphx::small_vector<int, 4>{1, 2});
    v.pop_back();
    v.pop_back();
    EXPECT(v.empty());
    EXPECT(v.capacity() == 4);
}

TEST_CASE(insert_single)
{
    migraphx::small_vector<int, 4> v{1, 3};
    auto* it = v.insert(v.begin() + 1, 2);
    EXPECT(*it == 2);
    EXPECT(it == v.begin() + 1);
    EXPECT(v == migraphx::small_vector<int, 4>{1, 2, 3});
    it = v.insert(v.begin(), 0);
    EXPECT(it == v.begin());
    EXPECT(v == migraphx::small_vector<int, 4>{0, 1, 2, 3});
    EXPECT(is_inline(v));
    // Grows while inserting in the middle
    it = v.insert(v.begin() + 2, 9);
    EXPECT(*it == 9);
    EXPECT(v == migraphx::small_vector<int, 4>{0, 1, 9, 2, 3});
    EXPECT(not is_inline(v));
    it = v.insert(v.end(), 4);
    EXPECT(it == v.end() - 1);
    EXPECT(v == migraphx::small_vector<int, 4>{0, 1, 9, 2, 3, 4});
}

TEST_CASE(insert_rvalue)
{
    migraphx::small_vector<std::unique_ptr<int>, 2> v;
    v.insert(v.end(), std::make_unique<int>(1));
    v.insert(v.begin(), std::make_unique<int>(0));
    v.insert(v.begin() + 1, std::make_unique<int>(5));
    EXPECT(v.size() == 3);
    EXPECT(*v[0] == 0);
    EXPECT(*v[1] == 5);
    EXPECT(*v[2] == 1);
}

TEST_CASE(insert_fill)
{
    migraphx::small_vector<int, 8> v{1, 2, 3, 4};
    // Tail longer than the insert
    auto* it = v.insert(v.begin() + 1, 2, 7);
    EXPECT(it == v.begin() + 1);
    EXPECT(v == migraphx::small_vector<int, 8>{1, 7, 7, 2, 3, 4});
    // Tail shorter than the insert
    it = v.insert(v.end() - 1, 2, 8);
    EXPECT(it == v.end() - 3);
    EXPECT(v == migraphx::small_vector<int, 8>{1, 7, 7, 2, 3, 8, 8, 4});
    EXPECT(is_inline(v));
    // Grows
    it = v.insert(v.begin() + 3, 3, 9);
    EXPECT(*it == 9);
    EXPECT(v == migraphx::small_vector<int, 8>{1, 7, 7, 9, 9, 9, 2, 3, 8, 8, 4});
    EXPECT(not is_inline(v));
    // Zero count is a no-op that returns pos
    it = v.insert(v.begin() + 2, 0, 1);
    EXPECT(it == v.begin() + 2);
    EXPECT(v.size() == 11);
}

TEST_CASE(insert_range)
{
    migraphx::small_vector<int, 8> v{1, 2, 3, 4};
    std::vector<int> a{7, 8};
    auto* it = v.insert(v.begin() + 1, a.begin(), a.end());
    EXPECT(it == v.begin() + 1);
    EXPECT(v == migraphx::small_vector<int, 8>{1, 7, 8, 2, 3, 4});
    std::list<int> b{5, 6, 7};
    it = v.insert(v.end() - 1, b.begin(), b.end());
    EXPECT(it == v.end() - 4);
    EXPECT(v == migraphx::small_vector<int, 8>{1, 7, 8, 2, 3, 5, 6, 7, 4});
    EXPECT(not is_inline(v));
    it = v.insert(v.end(), a.begin(), a.begin());
    EXPECT(it == v.end());
    EXPECT(v.size() == 9);
}

TEST_CASE(insert_input_range)
{
    migraphx::small_vector<int, 4> v{1, 2};
    std::istringstream ss("7 8 9");
    auto* it =
        v.insert(v.begin() + 1, std::istream_iterator<int>{ss}, std::istream_iterator<int>{});
    EXPECT(it == v.begin() + 1);
    EXPECT(v == migraphx::small_vector<int, 4>{1, 7, 8, 9, 2});
}

TEST_CASE(insert_initializer_list)
{
    migraphx::small_vector<int, 4> v{1, 4};
    auto* it = v.insert(v.begin() + 1, {2, 3});
    EXPECT(it == v.begin() + 1);
    EXPECT(v == migraphx::small_vector<int, 4>{1, 2, 3, 4});
}

TEST_CASE(insert_aliasing)
{
    migraphx::small_vector<std::string, 4> v{"a", "b", "c"};
    v.insert(v.begin(), v.back());
    EXPECT(v == migraphx::small_vector<std::string, 4>{"c", "a", "b", "c"});
    v.insert(v.begin() + 1, 2, v[2]);
    EXPECT(v == migraphx::small_vector<std::string, 4>{"c", "b", "b", "a", "b", "c"});
    v.insert(v.begin() + 1, v.front());
    EXPECT(v == migraphx::small_vector<std::string, 4>{"c", "c", "b", "b", "a", "b", "c"});
}

TEST_CASE(emplace_middle)
{
    migraphx::small_vector<std::string, 4> v{"a", "c"};
    auto* it = v.emplace(v.begin() + 1, 2, 'b');
    EXPECT(*it == "bb");
    EXPECT(v == migraphx::small_vector<std::string, 4>{"a", "bb", "c"});
    it = v.emplace(v.end(), "d");
    EXPECT(it == v.end() - 1);
    EXPECT(v == migraphx::small_vector<std::string, 4>{"a", "bb", "c", "d"});
    // Grows while emplacing an alias of an existing element
    it = v.emplace(v.begin(), v[1]);
    EXPECT(*it == "bb");
    EXPECT(v == migraphx::small_vector<std::string, 4>{"bb", "a", "bb", "c", "d"});
}

TEST_CASE(erase_single)
{
    migraphx::small_vector<int, 4> v{1, 2, 3, 4};
    auto* it = v.erase(v.begin() + 1);
    EXPECT(*it == 3);
    EXPECT(v == migraphx::small_vector<int, 4>{1, 3, 4});
    it = v.erase(v.end() - 1);
    EXPECT(it == v.end());
    EXPECT(v == migraphx::small_vector<int, 4>{1, 3});
    it = v.erase(v.begin());
    EXPECT(it == v.begin());
    EXPECT(v == migraphx::small_vector<int, 4>{3});
}

TEST_CASE(erase_range)
{
    migraphx::small_vector<int, 4> v{1, 2, 3, 4, 5, 6};
    auto* it = v.erase(v.begin() + 1, v.begin() + 3);
    EXPECT(*it == 4);
    EXPECT(v == migraphx::small_vector<int, 4>{1, 4, 5, 6});
    it = v.erase(v.begin() + 2, v.begin() + 2);
    EXPECT(it == v.begin() + 2);
    EXPECT(v == migraphx::small_vector<int, 4>{1, 4, 5, 6});
    it = v.erase(v.begin(), v.end());
    EXPECT(it == v.end());
    EXPECT(v.empty());
    EXPECT(v.capacity() == 6);
}

TEST_CASE(resize)
{
    migraphx::small_vector<int, 4> v{1, 2};
    v.resize(4);
    EXPECT(v == migraphx::small_vector<int, 4>{1, 2, 0, 0});
    EXPECT(is_inline(v));
    v.resize(6, 9);
    EXPECT(v == migraphx::small_vector<int, 4>{1, 2, 0, 0, 9, 9});
    EXPECT(not is_inline(v));
    v.resize(3);
    EXPECT(v == migraphx::small_vector<int, 4>{1, 2, 0});
    v.resize(3, 1);
    EXPECT(v == migraphx::small_vector<int, 4>{1, 2, 0});
    v.resize(0);
    EXPECT(v.empty());
    // The fill value may alias an element
    v = {5, 6};
    v.resize(7, v.front());
    EXPECT(v == migraphx::small_vector<int, 4>{5, 6, 5, 5, 5, 5, 5});
    v.resize(9, v.back());
    EXPECT(v == migraphx::small_vector<int, 4>{5, 6, 5, 5, 5, 5, 5, 5, 5});
}

TEST_CASE(swap_inline)
{
    migraphx::small_vector<std::string, 4> v{"a", "b", "c"};
    migraphx::small_vector<std::string, 4> w{"x"};
    v.swap(w);
    EXPECT(v == migraphx::small_vector<std::string, 4>{"x"});
    EXPECT(w == migraphx::small_vector<std::string, 4>{"a", "b", "c"});
    EXPECT(is_inline(v));
    EXPECT(is_inline(w));
    swap(v, w);
    EXPECT(v == migraphx::small_vector<std::string, 4>{"a", "b", "c"});
    EXPECT(w == migraphx::small_vector<std::string, 4>{"x"});
    migraphx::small_vector<std::string, 4> e;
    v.swap(e);
    EXPECT(v.empty());
    EXPECT(e == migraphx::small_vector<std::string, 4>{"a", "b", "c"});
    e.swap(e);
    EXPECT(e == migraphx::small_vector<std::string, 4>{"a", "b", "c"});
}

TEST_CASE(swap_heap)
{
    migraphx::small_vector<int, 2> v{1, 2, 3};
    migraphx::small_vector<int, 2> w{4, 5, 6, 7, 8};
    const int* vdata = v.data();
    const int* wdata = w.data();
    v.swap(w);
    EXPECT(v == migraphx::small_vector<int, 2>{4, 5, 6, 7, 8});
    EXPECT(w == migraphx::small_vector<int, 2>{1, 2, 3});
    EXPECT(v.data() == wdata);
    EXPECT(w.data() == vdata);
}

TEST_CASE(swap_mixed)
{
    migraphx::small_vector<std::string, 2> v{"a"};
    migraphx::small_vector<std::string, 2> w{"x", "y", "z"};
    const std::string* wdata = w.data();
    v.swap(w);
    EXPECT(v == migraphx::small_vector<std::string, 2>{"x", "y", "z"});
    EXPECT(w == migraphx::small_vector<std::string, 2>{"a"});
    EXPECT(v.data() == wdata);
    EXPECT(is_inline(w));
    w.swap(v);
    EXPECT(v == migraphx::small_vector<std::string, 2>{"a"});
    EXPECT(w == migraphx::small_vector<std::string, 2>{"x", "y", "z"});
    EXPECT(is_inline(v));
    EXPECT(w.data() == wdata);
}

TEST_CASE(comparisons)
{
    migraphx::small_vector<int, 4> a{1, 2, 3};
    migraphx::small_vector<int, 4> b{1, 2, 3};
    migraphx::small_vector<int, 4> c{1, 2, 4};
    migraphx::small_vector<int, 4> d{1, 2};
    EXPECT(a == b);
    EXPECT(not(a != b));
    EXPECT(a != c);
    EXPECT(a != d);
    EXPECT(a < c);
    EXPECT(d < a);
    EXPECT(not(a < b));
    EXPECT(a <= b);
    EXPECT(a <= c);
    EXPECT(c > a);
    EXPECT(a > d);
    EXPECT(a >= b);
    EXPECT(c >= a);
    EXPECT(not(d >= a));
}

TEST_CASE(comparisons_different_capacity)
{
    migraphx::small_vector<int, 2> a{1, 2, 3};
    migraphx::small_vector<int, 8> b{1, 2, 3};
    migraphx::small_vector<int, 8> c{1, 2, 3, 4};
    EXPECT(a == b);
    EXPECT(a != c);
    EXPECT(a < c);
    EXPECT(c > a);
    EXPECT(a <= b);
    EXPECT(a >= b);
}

TEST_CASE(move_only_type)
{
    migraphx::small_vector<std::unique_ptr<int>, 2> v;
    for(int i = 0; i < 5; i++)
        v.push_back(std::make_unique<int>(i));
    v.emplace_back(std::make_unique<int>(5));
    EXPECT(v.size() == 6);
    EXPECT(*v[3] == 3);
    v.erase(v.begin() + 1);
    EXPECT(*v[1] == 2);
    migraphx::small_vector<std::unique_ptr<int>, 2> w(std::move(v));
    EXPECT(w.size() == 5);
    EXPECT(*w.front() == 0);
    EXPECT(*w.back() == 5);
    v = std::move(w);
    EXPECT(v.size() == 5);
    v.shrink_to_fit();
    v.resize(2);
    EXPECT(*v[1] == 2);
    v.shrink_to_fit();
    EXPECT(is_inline(v));
    EXPECT(*v[1] == 2);
}

TEST_CASE(lifetime_tracking)
{
    EXPECT(counted::live == 0);
    {
        migraphx::small_vector<counted, 2> v{1, 2};
        EXPECT(counted::live == 2);
        v.push_back(3);
        v.insert(v.begin(), 0);
        EXPECT(counted::live == 4);
        v.erase(v.begin() + 1, v.begin() + 3);
        EXPECT(counted::live == 2);
        v.resize(6, 7);
        EXPECT(counted::live == 6);
        migraphx::small_vector<counted, 2> w(v);
        EXPECT(counted::live == 12);
        migraphx::small_vector<counted, 2> x(std::move(w));
        EXPECT(counted::live == 12);
        x.shrink_to_fit();
        x.resize(1);
        x.shrink_to_fit();
        EXPECT(counted::live == 7);
        v.swap(x);
        EXPECT(counted::live == 7);
        v.assign(3, 4);
        EXPECT(counted::live == 9);
        v.clear();
        EXPECT(counted::live == 6);
        x.pop_back();
        EXPECT(counted::live == 5);
        w = x;
        EXPECT(counted::live == 10);
    }
    EXPECT(counted::live == 0);
}

TEST_CASE(strong_exception_guarantee)
{
    throwing_copy::budget = 100;
    migraphx::small_vector<throwing_copy, 2> v{1, 2};
    EXPECT(v.capacity() == 2);
    // Growing copies the two existing elements, so throw on the second one
    throwing_copy::budget = 2;
    EXPECT(test::throws<std::runtime_error>([&] { v.push_back(3); }));
    throwing_copy::budget = 100;
    EXPECT(v == migraphx::small_vector<throwing_copy, 2>{1, 2});
    EXPECT(v.capacity() == 2);
    EXPECT(is_inline(v));
    v.push_back(3);
    EXPECT(v == migraphx::small_vector<throwing_copy, 2>{1, 2, 3});
    // Inserting in the middle while growing copies the two new elements first
    throwing_copy::budget = 3;
    EXPECT(test::throws<std::runtime_error>([&] { v.insert(v.begin() + 1, 2, 9); }));
    throwing_copy::budget = 100;
    EXPECT(v == migraphx::small_vector<throwing_copy, 2>{1, 2, 3});
}

TEST_CASE(string_elements)
{
    migraphx::small_vector<std::string> v;
    EXPECT(v.capacity() == 2);
    v.push_back("one");
    v.emplace_back("two");
    v.push_back("three");
    EXPECT(v.size() == 3);
    EXPECT(v.capacity() >= 3);
    EXPECT(v[2] == "three");
    migraphx::small_vector<std::string> w = v;
    w[0]                                  = "uno";
    EXPECT(v[0] == "one");
    EXPECT(w[0] == "uno");
    std::sort(w.begin(), w.end());
    EXPECT(w == migraphx::small_vector<std::string>{"three", "two", "uno"});
}

TEST_CASE(zero_inline_capacity)
{
    migraphx::small_vector<int, 0> v;
    EXPECT(v.empty());
    EXPECT(v.capacity() == 0);
    v.push_back(1);
    v.push_back(2);
    EXPECT(v == migraphx::small_vector<int, 0>{1, 2});
    EXPECT(v.capacity() >= 2);
    v.clear();
    v.shrink_to_fit();
    EXPECT(v.capacity() == 0);
    migraphx::small_vector<int, 0> w{3, 4, 5};
    v = std::move(w);
    EXPECT(v == migraphx::small_vector<int, 0>{3, 4, 5});
    EXPECT(w.empty()); // cppcheck-suppress accessMoved
}

TEST_CASE(algorithms)
{
    migraphx::small_vector<int, 4> v{5, 3, 9, 1, 7};
    std::sort(v.begin(), v.end());
    EXPECT(v == migraphx::small_vector<int, 4>{1, 3, 5, 7, 9});
    EXPECT(std::accumulate(v.begin(), v.end(), 0) == 25);
    auto* it = std::find(v.begin(), v.end(), 7);
    EXPECT(it == v.begin() + 3);
    EXPECT(std::binary_search(v.begin(), v.end(), 9));
    std::vector<int> out;
    std::copy(v.rbegin(), v.rend(), std::back_inserter(out));
    EXPECT(out == std::vector<int>{9, 7, 5, 3, 1});
    v.erase(std::remove_if(v.begin(), v.end(), [](int x) { return x > 4; }), v.end());
    EXPECT(v == migraphx::small_vector<int, 4>{1, 3});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
