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

#include <migraphx/transform_view.hpp>
#include <list>
#include <forward_list>
#include <vector>

#include <test.hpp>

TEST_CASE(basic_transform)
{
    std::vector<int> vec = {1, 2, 3, 4, 5};
    auto view            = migraphx::views::transform(vec, [](int x) { return x * x; });

    auto it = view.begin();
    EXPECT(it[0] == 1);
    EXPECT(it[1] == 4);
    EXPECT(it[2] == 9);
    EXPECT(it[3] == 16);
    EXPECT(it[4] == 25);

    EXPECT(*it == 1);
    it += 1;
    EXPECT(*it == 4);
    it += 1;
    EXPECT(*it == 9);
    it += 1;
    EXPECT(*it == 16);
    it += 1;
    EXPECT(*it == 25);

    auto it2 = view.end();
    it2 -= 1;
    EXPECT(*it2 == 25);
    it2 -= 1;
    EXPECT(*it2 == 16);
    it2 -= 1;
    EXPECT(*it2 == 9);
    it2 -= 1;
    EXPECT(*it2 == 4);
    it2 -= 1;
    EXPECT(*it2 == 1);
}

TEST_CASE(transform_const)
{
    std::vector<int> vec = {1, 2, 3, 4, 5};
    const auto& cvec     = vec;
    auto view            = migraphx::views::transform(cvec, [](int x) { return x * x; });

    auto it = view.begin();
    EXPECT(it[0] == 1);
    EXPECT(it[1] == 4);
    EXPECT(it[2] == 9);
    EXPECT(it[3] == 16);
    EXPECT(it[4] == 25);

    EXPECT(*it == 1);
    it += 1;
    EXPECT(*it == 4);
    it += 1;
    EXPECT(*it == 9);
    it += 1;
    EXPECT(*it == 16);
    it += 1;
    EXPECT(*it == 25);

    auto it2 = view.end();
    it2 -= 1;
    EXPECT(*it2 == 25);
    it2 -= 1;
    EXPECT(*it2 == 16);
    it2 -= 1;
    EXPECT(*it2 == 9);
    it2 -= 1;
    EXPECT(*it2 == 4);
    it2 -= 1;
    EXPECT(*it2 == 1);
}

TEST_CASE(transform_with_reference)
{
    std::vector<int> vec = {1, 2, 3, 4, 5};
    // cppcheck-suppress constParameterReference
    auto view = migraphx::views::transform(vec, [](int& x) -> int& { return x; });

    auto it = view.begin();
    EXPECT(*it == 1);
    ++it;
    EXPECT(*it == 2);
    ++it;
    EXPECT(*it == 3);
    ++it;
    EXPECT(*it == 4);
    ++it;
    EXPECT(*it == 5);

    auto it2 = view.end();
    --it2;
    EXPECT(*it2 == 5);
    --it2;
    EXPECT(*it2 == 4);
    --it2;
    EXPECT(*it2 == 3);
    --it2;
    EXPECT(*it2 == 2);
    --it2;
    EXPECT(*it2 == 1);

    // Modify the original vector through the view
    *view.begin() = 10;
    EXPECT(vec[0] == 10);
}

TEST_CASE(empty_range)
{
    std::vector<int> vec;
    auto view = migraphx::views::transform(vec, [](int x) { return x * x; });

    EXPECT(view.begin() == view.end());
}

TEST_CASE(non_random_access_iterator)
{
    std::list<int> lst = {1, 2, 3, 4, 5};
    auto view          = migraphx::views::transform(lst, [](int x) { return x * 2; });

    auto it = view.begin();
    EXPECT(*it == 2);
    ++it;
    EXPECT(*it == 4);
    ++it;
    EXPECT(*it == 6);
    ++it;
    EXPECT(*it == 8);
    ++it;
    EXPECT(*it == 10);
}

TEST_CASE(non_random_access_iterator_with_reference)
{
    std::list<int> lst = {1, 2, 3, 4, 5};
    // cppcheck-suppress constParameterReference
    auto view = migraphx::views::transform(lst, [](int& x) -> int& { return x; });

    auto it = view.begin();
    EXPECT(*it == 1);
    ++it;
    EXPECT(*it == 2);
    ++it;
    EXPECT(*it == 3);
    ++it;
    EXPECT(*it == 4);
    ++it;
    EXPECT(*it == 5);

    // Modify the original list through the view
    *view.begin() = 10;
    EXPECT(lst.front() == 10);
}

TEST_CASE(forward_iterator)
{
    std::forward_list<int> flst = {1, 2, 3, 4, 5};
    auto view                   = migraphx::views::transform(flst, [](int x) { return x + 1; });

    auto it = view.begin();
    EXPECT(*it == 2);
    ++it;
    EXPECT(*it == 3);
    ++it;
    EXPECT(*it == 4);
    ++it;
    EXPECT(*it == 5);
    ++it;
    EXPECT(*it == 6);

    auto it2 = view.begin();
    std::advance(it2, 3);
    EXPECT(*it2 == 5);
}

TEST_CASE(forward_iterator_with_reference)
{
    std::forward_list<int> flst = {1, 2, 3, 4, 5};
    // cppcheck-suppress constParameterReference
    auto view = migraphx::views::transform(flst, [](int& x) -> int& { return x; });

    auto it = view.begin();
    EXPECT(*it == 1);
    ++it;
    EXPECT(*it == 2);
    ++it;
    EXPECT(*it == 3);
    ++it;
    EXPECT(*it == 4);
    ++it;
    EXPECT(*it == 5);

    // Modify the original forward_list through the view
    *view.begin() = 10;
    EXPECT(flst.front() == 10);
}

TEST_CASE(transform_view_element_comparison)
{
    std::vector<int> vec1 = {1, 2, 3, 4, 5};
    std::vector<int> vec2 = {1, 2, 3, 4, 5};
    std::vector<int> vec3 = {5, 4, 3, 2, 1};

    auto squared = [](int x) { return x * x; };

    auto view1 = migraphx::views::transform(vec1, squared);
    auto view2 = migraphx::views::transform(vec2, squared);
    auto view3 = migraphx::views::transform(vec3, squared);

    EXPECT(view1 == view2); // Same elements
    EXPECT(view1 != view3); // Different elements
    EXPECT(view1 < view3);  // Lexicographical comparison
    EXPECT(view1 <= view3);
    EXPECT(view3 > view1);
    EXPECT(view3 >= view1);
}

TEST_CASE(transform_view_element_comparison_diff_view)
{
    std::vector<int> vec1 = {1, 2, 3, 4, 5};
    std::vector<int> vec2 = {2, 3, 4, 5, 6};

    auto view1 = migraphx::views::transform(vec1, [](int x) { return x * x; });
    auto view2 = migraphx::views::transform(vec2, [](int x) { return (x - 1) * (x - 1); });
    auto view3 = migraphx::views::transform(vec2, [](int x) { return x * x; });

    EXPECT(view1 == view2);
    EXPECT(view1 != view3);
    EXPECT(view2 != view3);

    EXPECT(view1 < view3);
    EXPECT(view2 < view3);
    EXPECT(view1 <= view3);
    EXPECT(view2 <= view3);
    EXPECT(view3 > view1);
    EXPECT(view3 > view2);
    EXPECT(view3 >= view1);
    EXPECT(view3 >= view2);
}

struct non_comparable
{
    int value;

    friend bool operator==(const non_comparable& lhs, const non_comparable& rhs)
    {
        return lhs.value == rhs.value;
    }

    friend bool operator!=(const non_comparable& lhs, const non_comparable& rhs)
    {
        return not(lhs == rhs);
    }
};

TEST_CASE(transform_view_non_comparable_elements)
{

    std::vector<int> vec1  = {1, 2, 3};
    std::vector<int> vec2  = {2, 3, 4};
    auto as_non_comparable = [](int x) -> non_comparable { return {x}; };
    auto view              = migraphx::views::transform(vec1, as_non_comparable);
    auto view2             = migraphx::views::transform(vec1, as_non_comparable);
    auto view3             = migraphx::views::transform(vec2, as_non_comparable);

    EXPECT(view == view2);
    EXPECT(view != view3);
}

TEST_CASE(operator_arrow_in_loop_reference)
{
    struct a
    {
        int val;
    };
    std::vector<a> data{{1}, {2}, {3}};
    auto view = migraphx::views::transform(data, [](const a& t) -> const a& { return t; });
    int sum   = 0;
    for(auto it = view.begin(); it != view.end(); ++it)
    {
        sum += it->val;
    }
    EXPECT(sum == 6);
}

TEST_CASE(operator_arrow_in_loop_value)
{
    struct a
    {
        int val;
    };
    std::vector<a> data{{1}, {2}, {3}};
    auto view = migraphx::views::transform(data, [](const a& t) { return a{t.val * 2}; });
    std::vector<int> out;
    for(auto it = view.begin(); it != view.end(); ++it)
    {
        out.push_back(it->val);
    }
    EXPECT(out.size() == 3);
    EXPECT(out[0] == 2);
    EXPECT(out[1] == 4);
    EXPECT(out[2] == 6);
}

TEST_CASE(transform_view_mutate_member)
{
    struct a
    {
        int val;
    };
    std::vector<a> data{{1}, {2}, {3}};
    // cppcheck-suppress constParameterReference
    auto view = migraphx::views::transform(data, [](auto& t) -> auto& { return t.val; });
    std::for_each(view.begin(), view.end(), [](auto& i) { i++; });
    std::vector<a> edata{{2}, {3}, {4}};
    EXPECT(std::equal(data.begin(), data.end(), edata.begin(), [](const a& lhs, const a& rhs) {
        return lhs.val == rhs.val;
    }));
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
