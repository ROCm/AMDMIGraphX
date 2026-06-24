/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <migraphx/zip_view.hpp>
#include <migraphx/float_equal.hpp>
#include <algorithm>
#include <forward_list>
#include <iterator>
#include <list>
#include <numeric>
#include <tuple>
#include <vector>

#include <test.hpp>

TEST_CASE(basic_zip)
{
    std::vector<int> a  = {1, 2, 3, 4, 5};
    std::vector<char> b = {'a', 'b', 'c', 'd', 'e'};
    auto view           = migraphx::views::zip(a, b);

    auto it = view.begin();
    EXPECT(std::get<0>(it[0]) == 1);
    EXPECT(std::get<1>(it[0]) == 'a');
    EXPECT(std::get<0>(it[4]) == 5);
    EXPECT(std::get<1>(it[4]) == 'e');

    EXPECT(std::get<0>(*it) == 1);
    EXPECT(std::get<1>(*it) == 'a');
    it += 1;
    EXPECT(std::get<0>(*it) == 2);
    EXPECT(std::get<1>(*it) == 'b');
    it += 3;
    EXPECT(std::get<0>(*it) == 5);
    EXPECT(std::get<1>(*it) == 'e');

    auto it2 = view.end();
    it2 -= 1;
    EXPECT(std::get<0>(*it2) == 5);
    it2 -= 4;
    EXPECT(std::get<0>(*it2) == 1);

    EXPECT((view.end() - view.begin()) == 5);
}

TEST_CASE(zip_iterator_ordering)
{
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {4, 5, 6};
    auto view          = migraphx::views::zip(a, b);

    auto first = view.begin();
    auto last  = view.end();
    EXPECT(first < last);
    EXPECT(last > first);
    EXPECT(first <= first);
    EXPECT(first != last);
    EXPECT((first + 3) == last);
}

TEST_CASE(zip_three_ranges)
{
    std::vector<int> a    = {1, 2};
    std::vector<double> b = {1.5, 2.5};
    std::vector<char> c   = {'x', 'y'};
    auto view             = migraphx::views::zip(a, b, c);

    auto it = view.begin();
    EXPECT(std::get<0>(*it) == 1);
    EXPECT(migraphx::float_equal(std::get<1>(*it), 1.5));
    EXPECT(std::get<2>(*it) == 'x');
    ++it;
    EXPECT(std::get<0>(*it) == 2);
    EXPECT(migraphx::float_equal(std::get<1>(*it), 2.5));
    EXPECT(std::get<2>(*it) == 'y');
    ++it;
    EXPECT(it == view.end());
}

TEST_CASE(zip_const)
{
    const std::vector<int> a  = {1, 2, 3};
    const std::vector<char> b = {'a', 'b', 'c'};
    auto view                 = migraphx::views::zip(a, b);

    auto it = view.begin();
    EXPECT(std::get<0>(*it) == 1);
    EXPECT(std::get<1>(*it) == 'a');
    ++it;
    EXPECT(std::get<0>(*it) == 2);
    EXPECT(std::get<1>(*it) == 'b');
    ++it;
    EXPECT(std::get<0>(*it) == 3);
    EXPECT(std::get<1>(*it) == 'c');
}

TEST_CASE(zip_mutate)
{
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {4, 5, 6};
    auto view          = migraphx::views::zip(a, b);

    for(auto&& t : view)
    {
        std::get<0>(t) += 10;
        std::get<1>(t) *= 2;
    }

    EXPECT(a == std::vector<int>({11, 12, 13}));
    EXPECT(b == std::vector<int>({8, 10, 12}));
}

TEST_CASE(zip_different_lengths)
{
    std::vector<int> a = {1, 2, 3, 4, 5};
    std::vector<int> b = {10, 20, 30};
    auto view          = migraphx::views::zip(a, b);

    // The zip stops at the shortest range.
    EXPECT((view.end() - view.begin()) == 3);

    std::vector<int> seen;
    std::transform(
        view.begin(), view.end(), std::back_inserter(seen), [](auto t) { return std::get<0>(t); });
    EXPECT(seen == std::vector<int>({1, 2, 3}));
}

TEST_CASE(zip_different_lengths_first_shorter)
{
    std::vector<int> a = {1, 2};
    std::vector<int> b = {10, 20, 30, 40};
    auto view          = migraphx::views::zip(a, b);

    EXPECT((view.end() - view.begin()) == 2);

    std::vector<int> seen;
    std::transform(
        view.begin(), view.end(), std::back_inserter(seen), [](auto t) { return std::get<1>(t); });
    EXPECT(seen == std::vector<int>({10, 20}));
}

TEST_CASE(zip_empty)
{
    std::vector<int> a;
    std::vector<int> b = {1, 2, 3};
    auto view          = migraphx::views::zip(a, b);

    EXPECT(view.begin() == view.end());
    EXPECT((view.end() - view.begin()) == 0);
}

TEST_CASE(zip_bidirectional_iterator)
{
    std::list<int> a   = {1, 2, 3, 4};
    std::vector<int> b = {5, 6, 7, 8};
    auto view          = migraphx::views::zip(a, b);

    auto it = view.begin();
    EXPECT(std::get<0>(*it) == 1);
    EXPECT(std::get<1>(*it) == 5);
    ++it;
    EXPECT(std::get<0>(*it) == 2);
    EXPECT(std::get<1>(*it) == 6);
    --it;
    EXPECT(std::get<0>(*it) == 1);
    EXPECT(std::get<1>(*it) == 5);
}

TEST_CASE(zip_bidirectional_mutate)
{
    std::list<int> a   = {1, 2, 3};
    std::vector<int> b = {4, 5, 6};
    auto view          = migraphx::views::zip(a, b);

    for(auto&& t : view)
        std::get<0>(t) += std::get<1>(t);

    EXPECT(a == std::list<int>({5, 7, 9}));
}

TEST_CASE(zip_forward_iterator)
{
    std::forward_list<int> a = {1, 2, 3, 4};
    std::vector<char> b      = {'a', 'b', 'c', 'd'};
    auto view                = migraphx::views::zip(a, b);

    std::vector<int> ints;
    std::vector<char> chars;
    for(auto&& t : view)
    {
        ints.push_back(std::get<0>(t));
        chars.push_back(std::get<1>(t));
    }
    EXPECT(ints == std::vector<int>({1, 2, 3, 4}));
    EXPECT(chars == std::vector<char>({'a', 'b', 'c', 'd'}));
}

TEST_CASE(zip_view_comparison)
{
    std::vector<int> a1 = {1, 2, 3};
    std::vector<int> a2 = {1, 2, 3};
    std::vector<int> a3 = {1, 2, 4};
    std::vector<char> b = {'x', 'y', 'z'};

    auto view1 = migraphx::views::zip(a1, b);
    auto view2 = migraphx::views::zip(a2, b);
    auto view3 = migraphx::views::zip(a3, b);

    EXPECT(view1 == view2); // Same elements
    EXPECT(view1 != view3); // Different elements
    EXPECT(view1 < view3);  // Lexicographical comparison
    EXPECT(view1 <= view3);
    EXPECT(view3 > view1);
    EXPECT(view3 >= view1);
}

TEST_CASE(zip_with_algorithm)
{
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {4, 5, 6};
    auto view          = migraphx::views::zip(a, b);

    int dot = std::accumulate(view.begin(), view.end(), 0, [](int acc, auto t) {
        return acc + std::get<0>(t) * std::get<1>(t);
    });
    EXPECT(dot == 32); // 1*4 + 2*5 + 3*6
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
