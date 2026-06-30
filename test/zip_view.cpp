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
#include <type_traits>
#include <utility>
#include <vector>

#include <test.hpp>

template <class It, class = void>
struct has_predecrement : std::false_type
{
};
template <class It>
struct has_predecrement<It, std::void_t<decltype(--std::declval<It&>())>> : std::true_type
{
};

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

TEST_CASE(zip_bidirectional_forward_only)
{
    // A bidirectional range (list) caps the zip at a forward iterator: no decrement.
    std::list<int> a   = {1, 2, 3, 4};
    std::vector<int> b = {5, 6, 7, 8};
    auto view          = migraphx::views::zip(a, b);

    using it_type = decltype(view.begin());
    static_assert(std::is_same<it_type::iterator_category, std::forward_iterator_tag>{},
                  "a bidirectional underlying range should produce a forward zip iterator");
    static_assert(not has_predecrement<it_type>{},
                  "a forward zip iterator must not be decrementable");

    auto it = view.begin();
    EXPECT(std::get<0>(*it) == 1);
    EXPECT(std::get<1>(*it) == 5);
    ++it;
    EXPECT(std::get<0>(*it) == 2);
    EXPECT(std::get<1>(*it) == 6);
    ++it;
    EXPECT(std::get<0>(*it) == 3);
    EXPECT(std::get<1>(*it) == 7);
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

TEST_CASE(zip_forward_different_lengths)
{
    // Forward range governs the zip; stopping at the shortest range relies on `or` equality.
    std::forward_list<int> a = {1, 2, 3};
    std::vector<int> b       = {10, 20, 30, 40, 50};
    auto view                = migraphx::views::zip(a, b);

    std::vector<int> seen0;
    std::vector<int> seen1;
    for(auto&& t : view)
    {
        seen0.push_back(std::get<0>(t));
        seen1.push_back(std::get<1>(t));
    }
    EXPECT(seen0 == std::vector<int>({1, 2, 3}));
    EXPECT(seen1 == std::vector<int>({10, 20, 30}));
}

TEST_CASE(zip_bidirectional_different_lengths)
{
    // Forward iteration over unequal-length bidirectional ranges stops at the shortest range.
    std::list<int> a = {1, 2, 3, 4, 5};
    std::list<int> b = {10, 20, 30};
    auto view        = migraphx::views::zip(a, b);

    std::vector<int> seen0;
    std::vector<int> seen1;
    for(auto&& t : view)
    {
        seen0.push_back(std::get<0>(t));
        seen1.push_back(std::get<1>(t));
    }
    EXPECT(seen0 == std::vector<int>({1, 2, 3}));
    EXPECT(seen1 == std::vector<int>({10, 20, 30}));
}

TEST_CASE(zip_random_access_reverse)
{
    std::vector<int> a = {1, 2, 3};
    std::vector<int> b = {4, 5, 6};
    auto view          = migraphx::views::zip(a, b);

    using it_type = decltype(view.begin());
    static_assert(std::is_same<it_type::iterator_category, std::random_access_iterator_tag>{},
                  "random-access underlying ranges should produce a random-access zip iterator");
    static_assert(has_predecrement<it_type>{}, "a random-access zip iterator must be decrementable");

    std::vector<int> seen0;
    std::vector<int> seen1;
    auto it = view.end();
    while(it != view.begin())
    {
        --it;
        seen0.push_back(std::get<0>(*it));
        seen1.push_back(std::get<1>(*it));
    }
    EXPECT(seen0 == std::vector<int>({3, 2, 1}));
    EXPECT(seen1 == std::vector<int>({6, 5, 4}));
}

TEST_CASE(zip_reverse_matches_forward_random_access)
{
    // Truncated end makes reverse traversal match forward (reversed), even for unequal lengths.
    std::vector<int> a = {1, 2, 3, 4, 5};
    std::vector<int> b = {10, 20, 30};
    auto view          = migraphx::views::zip(a, b);

    std::vector<int> fwd0;
    std::vector<int> fwd1;
    for(auto&& t : view)
    {
        fwd0.push_back(std::get<0>(t));
        fwd1.push_back(std::get<1>(t));
    }

    std::vector<int> bwd0;
    std::vector<int> bwd1;
    auto it = view.end();
    while(it != view.begin())
    {
        --it;
        bwd0.push_back(std::get<0>(*it));
        bwd1.push_back(std::get<1>(*it));
    }
    std::reverse(bwd0.begin(), bwd0.end());
    std::reverse(bwd1.begin(), bwd1.end());

    EXPECT(fwd0 == std::vector<int>({1, 2, 3}));
    EXPECT(fwd1 == std::vector<int>({10, 20, 30}));
    EXPECT(fwd0 == bwd0);
    EXPECT(fwd1 == bwd1);
}

TEST_CASE(zip_random_access_ordering_consistency)
{
    // == must stay consistent with <: the shortest-length iterator equals end() and is
    // neither less nor greater.
    std::vector<int> a = {1, 2, 3, 4, 5};
    std::vector<int> b = {10, 20, 30};
    auto view          = migraphx::views::zip(a, b);

    auto first         = view.begin();
    auto last          = view.end();
    const std::ptrdiff_t shortest = 3;

    EXPECT((last - first) == shortest);
    EXPECT(first < last);
    EXPECT(first != last);

    auto mid = first + shortest;
    EXPECT(mid == last);
    EXPECT(not(mid < last));
    EXPECT(not(mid > last));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
