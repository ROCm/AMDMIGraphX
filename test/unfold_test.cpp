/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *
 */
#include <migraphx/unfold.hpp>

#include <algorithm>
#include <memory>
#include <numeric>
#include <list>
#include <vector>
#include <test.hpp>

using migraphx::unfold;

TEST_CASE(test_basic_sequence)
{
    auto f = [](int x) { return x; };
    auto g = [](int x) -> std::optional<int> {
        return (x < 5) ? std::optional{x + 1} : std::nullopt;
    };
    auto rng = unfold(1, f, g);

    std::vector<int> got;
    std::copy(rng.begin(), rng.end(), std::back_inserter(got));

    EXPECT(got.size() == 5);
    EXPECT(got == std::vector<int>{1, 2, 3, 4, 5});
}

TEST_CASE(test_empty_sequence)
{
    auto f   = [](int x) { return x; };
    auto g   = [](auto) { return std::nullopt; };
    auto rng = unfold<int>(std::nullopt, f, g);

    auto it = rng.begin();
    EXPECT(bool{it == rng.end()});
}

TEST_CASE(test_one_element_sequence)
{
    auto f   = [](int x) { return x * x; };
    auto g   = [](int) -> std::optional<int> { return std::nullopt; };
    auto rng = unfold(7, f, g);

    auto it = rng.begin();
    EXPECT(bool{it != rng.end()});
    EXPECT(*it == 49);
    ++it;
    EXPECT(bool{it == rng.end()});
}

TEST_CASE(test_pair_state)
{
    using state = std::pair<int, int>;
    auto f      = [](const state& s) { return s.first + s.second; };
    auto g      = [](const state& s) -> std::optional<state> {
        if(s.first > 0)
            return state{s.first - 1, s.second + 2};
        return std::nullopt;
    };
    auto rng = unfold(state{3, 10}, f, g);

    std::vector<int> got;
    std::copy(rng.begin(), rng.end(), std::back_inserter(got));

    EXPECT(got == std::vector<int>{13, 14, 15, 16});
}

TEST_CASE(test_string_accumulation)
{
    auto f = [](const std::string& s) { return s; };
    auto g = [](const std::string& s) -> std::optional<std::string> {
        if(s.size() < 4)
            return s + "a";
        return std::nullopt;
    };
    auto rng = unfold(std::string("b"), f, g);

    std::vector<std::string> got;
    std::copy(rng.begin(), rng.end(), std::back_inserter(got));
    EXPECT(got == std::vector<std::string>{"b", "ba", "baa", "baaa"});
}

TEST_CASE(test_string_accumulation_auto)
{
    auto g = [](const std::string& s) -> std::optional<std::string> {
        if(s.size() < 4)
            return s + "a";
        return std::nullopt;
    };
    auto rng = unfold(std::string("b"), g);

    std::vector<std::string> got;
    std::copy(rng.begin(), rng.end(), std::back_inserter(got));
    EXPECT(got == std::vector<std::string>{"b", "ba", "baa", "baaa"});
}

TEST_CASE(test_iterator_semantics)
{
    auto f = [](int x) { return x * 2; };
    auto g = [](int x) -> std::optional<int> {
        return (x < 4) ? std::make_optional(x + 1) : std::nullopt;
    };
    auto rng = unfold(2, f, g);

    auto it = rng.begin();
    EXPECT(bool{it != rng.end()});
    EXPECT(*it == 4);
    auto it2 = it++;
    EXPECT(*it2 == 4);
    EXPECT(*it == 6);
    ++it;
    EXPECT(*it == 8);
    ++it;
    EXPECT(bool{it == rng.end()});
}

TEST_CASE(test_multiple_iterators)
{
    auto f = [](int x) { return x; };
    auto g = [](int x) -> std::optional<int> {
        return (x < 3) ? std::make_optional(x + 1) : std::nullopt;
    };
    auto rng = unfold(1, f, g);

    auto it1 = rng.begin();
    auto it2 = it1;
    EXPECT(*it1 == 1);
    ++it1;
    EXPECT(*it1 == 2);
    EXPECT(*it2 == 1);
    ++it2;
    ++it2;
    EXPECT(*it2 == 3);
    ++it2;
    EXPECT(bool{it2 == rng.end()});
}

TEST_CASE(test_large_sequence)
{
    auto f = [](int x) { return x; };
    auto g = [](int x) -> std::optional<int> {
        return (x < 10000) ? std::make_optional(x + 1) : std::nullopt;
    };
    auto rng = unfold(0, f, g);

    int count = std::distance(rng.begin(), rng.end());
    EXPECT(count == 10001);
}

TEST_CASE(test_squares)
{
    auto f = [](int x) { return x * x; };
    auto g = [](int x) -> std::optional<int> {
        return (x < 6) ? std::make_optional(x + 1) : std::nullopt;
    };
    auto rng = unfold(1, f, g);

    std::vector<int> got;
    std::copy(rng.begin(), rng.end(), std::back_inserter(got));
    EXPECT(got == std::vector<int>{1, 4, 9, 16, 25, 36});
}

TEST_CASE(test_move_only_function)
{
    auto ptr = std::make_unique<int>(10);
    auto f   = [p = std::move(ptr)](int x) { return x + *p; };
    auto g   = [](int x) -> std::optional<int> {
        return (x < 3) ? std::make_optional(x + 1) : std::nullopt;
    };
    auto rng = unfold(1, std::move(f), g);

    std::vector<int> got;
    std::copy(rng.begin(), rng.end(), std::back_inserter(got));
    EXPECT(got == std::vector<int>{11, 12, 13});
}

TEST_CASE(test_std_accumulate)
{
    auto f = [](int x) { return x; };
    auto g = [](int x) -> std::optional<int> {
        return (x < 5) ? std::make_optional(x + 1) : std::nullopt;
    };
    auto rng = unfold(1, f, g);

    int sum = std::accumulate(rng.begin(), rng.end(), 0);
    EXPECT(sum == 15);
}

TEST_CASE(test_std_copy_to_list)
{
    auto f = [](int x) { return x * 2; };
    auto g = [](int x) -> std::optional<int> {
        return (x < 4) ? std::make_optional(x + 1) : std::nullopt;
    };
    auto rng = unfold(1, f, g);

    std::list<int> result;
    std::copy(rng.begin(), rng.end(), std::back_inserter(result));
    EXPECT(result == std::list<int>{2, 4, 6, 8});
}

TEST_CASE(test_pointer_state)
{
    int arr[] = {1, 2, 3, 4, 5, 0};
    auto f    = [](const int* p) { return *p; };
    auto g    = [](const int* p) -> std::optional<const int*> {
        return (*(p + 1) == 0) ? std::nullopt : std::optional{p + 1};
    };
    auto rng = unfold(static_cast<const int*>(arr), f, g);

    std::vector<int> got;
    std::copy(rng.begin(), rng.end(), std::back_inserter(got));
    EXPECT(got == std::vector<int>{1, 2, 3, 4, 5});
}

TEST_CASE(test_infinite_sequence)
{
    auto f   = [](int x) { return x; };
    auto g   = [](int x) -> std::optional<int> { return x + 1; };
    auto rng = unfold(0, f, g);

    auto it = rng.begin();
    std::advance(it, 1000);
    EXPECT(*it == 1000);
}

TEST_CASE(test_floating_point)
{
    auto f = [](double x) { return x; };
    auto g = [](double x) -> std::optional<double> {
        return (x < 1.5) ? std::optional<double>{x + 0.5} : std::nullopt;
    };
    auto rng = unfold(0.0, f, g);

    std::vector<double> got;
    std::copy(rng.begin(), rng.end(), std::back_inserter(got));
    EXPECT(got.size() == 4);
    EXPECT(test::within_abs(got[0], 0.0) and test::within_abs(got[1], 0.5) and
           test::within_abs(got[2], 1.0) and test::within_abs(got[3], 1.5));
}

struct custom_struct_state
{
    int v;
    friend bool operator==(const custom_struct_state& x, const custom_struct_state& y)
    {
        return x.v == y.v;
    }
    friend bool operator!=(const custom_struct_state& x, const custom_struct_state& y)
    {
        return x.v != y.v;
    }
};
TEST_CASE(test_custom_struct_state)
{
    auto f = [](const custom_struct_state& c) { return c.v * 3; };
    auto g = [](const custom_struct_state& c) -> std::optional<custom_struct_state> {
        return (c.v < 4) ? std::make_optional(custom_struct_state{c.v + 1}) : std::nullopt;
    };
    auto rng = unfold(custom_struct_state{1}, f, g);

    std::vector<int> got;
    std::copy(rng.begin(), rng.end(), std::back_inserter(got));
    EXPECT(got == std::vector<int>{3, 6, 9, 12});
}

TEST_CASE(test_string_state)
{
    auto f = [](const std::string& s) { return s.size(); };
    auto g = [](const std::string& s) -> std::optional<std::string> {
        if(s.size() < 3)
            return s + "!";
        return std::nullopt;
    };
    auto rng = unfold(std::string("A"), f, g);

    std::vector<std::size_t> got;
    std::copy(rng.begin(), rng.end(), std::back_inserter(got));
    EXPECT(got == std::vector<std::size_t>{1, 2, 3});
}

TEST_CASE(test_even_numbers)
{
    auto f = [](int x) { return x; };
    auto g = [](int x) -> std::optional<int> {
        if(x < 10)
            return x + 2;
        return std::nullopt;
    };
    auto rng = unfold(0, f, g);

    std::vector<int> got;
    std::copy(rng.begin(), rng.end(), std::back_inserter(got));
    EXPECT(got == std::vector<int>{0, 2, 4, 6, 8, 10});
}

TEST_CASE(test_fibonacci_sequence)
{
    using state = std::pair<int, int>;
    auto f      = [](const state& s) { return s.first; };
    auto g      = [](const state& s) -> std::optional<state> {
        if(s.first > 100)
            return std::nullopt;
        return state{s.second, s.first + s.second};
    };
    auto rng = unfold(state{0, 1}, f, g);

    std::vector<int> got;
    std::copy(rng.begin(), rng.end(), std::back_inserter(got));
    std::vector<int> expected{0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144};
    got.resize(expected.size());
    EXPECT(got == expected);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
