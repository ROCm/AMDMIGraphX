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
 *
 */
#include <migraphx/par.hpp>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <test.hpp>

// par_transform unary tests

TEST_CASE(par_transform_unary_basic)
{
    std::vector<int> input = {1, 2, 3, 4, 5};
    std::vector<int> output(input.size());
    auto it = migraphx::par_transform(
        input.begin(), input.end(), output.begin(), [](int x) { return x * 2; });
    EXPECT(output == std::vector<int>{2, 4, 6, 8, 10});
    EXPECT(it == output.end());
    EXPECT(*(it - 1) == 10);
}

TEST_CASE(par_transform_unary_empty)
{
    std::vector<int> input;
    std::vector<int> output;
    auto it = migraphx::par_transform(
        input.begin(), input.end(), output.begin(), [](int x) { return x; });
    EXPECT(it == output.begin());
    EXPECT(output.empty());
}

TEST_CASE(par_transform_unary_single)
{
    std::vector<int> input = {42};
    std::vector<int> output(1);
    auto it = migraphx::par_transform(
        input.begin(), input.end(), output.begin(), [](int x) { return x + 1; });
    EXPECT(output == std::vector<int>{43});
    EXPECT(it == output.end());
    EXPECT(*(it - 1) == 43);
}

TEST_CASE(par_transform_unary_type_change)
{
    std::vector<int> input = {1, 4, 9, 16};
    std::vector<double> output(input.size());
    auto it = migraphx::par_transform(input.begin(), input.end(), output.begin(), [](int x) {
        return static_cast<double>(x) + 0.5;
    });
    EXPECT(output == std::vector<double>{1.5, 4.5, 9.5, 16.5});
    EXPECT(it == output.end());
    EXPECT(*(it - 1) == 16.5);
}

TEST_CASE(par_transform_unary_return_iterator)
{
    std::vector<int> input = {1, 2, 3};
    std::vector<int> output(input.size());
    auto it = migraphx::par_transform(
        input.begin(), input.end(), output.begin(), [](int x) { return x; });
    EXPECT(it == output.end());
    EXPECT(*(it - 1) == 3);
}

// par_transform binary tests

TEST_CASE(par_transform_binary_basic)
{
    std::vector<int> a = {1, 2, 3, 4};
    std::vector<int> b = {10, 20, 30, 40};
    std::vector<int> output(a.size());
    auto it = migraphx::par_transform(
        a.begin(), a.end(), b.begin(), output.begin(), [](int x, int y) { return x + y; });
    EXPECT(output == std::vector<int>{11, 22, 33, 44});
    EXPECT(it == output.end());
    EXPECT(*(it - 1) == 44);
}

TEST_CASE(par_transform_binary_empty)
{
    std::vector<int> a;
    std::vector<int> b;
    std::vector<int> output;
    auto it = migraphx::par_transform(
        a.begin(), a.end(), b.begin(), output.begin(), [](int x, int y) { return x + y; });
    EXPECT(it == output.begin());
    EXPECT(output.empty());
}

TEST_CASE(par_transform_binary_multiply)
{
    std::vector<int> a = {2, 3, 4};
    std::vector<int> b = {5, 6, 7};
    std::vector<int> output(a.size());
    auto it = migraphx::par_transform(
        a.begin(), a.end(), b.begin(), output.begin(), [](int x, int y) { return x * y; });
    EXPECT(output == std::vector<int>{10, 18, 28});
    EXPECT(it == output.end());
    EXPECT(*(it - 1) == 28);
}

TEST_CASE(par_transform_binary_return_iterator)
{
    std::vector<int> a = {1, 2};
    std::vector<int> b = {3, 4};
    std::vector<int> output(a.size());
    auto it = migraphx::par_transform(
        a.begin(), a.end(), b.begin(), output.begin(), [](int x, int y) { return x + y; });
    EXPECT(it == output.end());
    EXPECT(*(it - 1) == 6);
}

// par_for_each tests

TEST_CASE(par_for_each_basic)
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    std::vector<int> expected(v.size());
    std::transform(v.begin(), v.end(), expected.begin(), [](int x) { return x * 3; });
    migraphx::par_for_each(v.begin(), v.end(), [](int& x) { x *= 3; });
    EXPECT(v == expected);
}

TEST_CASE(par_for_each_empty)
{
    std::vector<int> v;
    migraphx::par_for_each(v.begin(), v.end(), [](int& x) { x += 1; });
    EXPECT(v.empty());
}

TEST_CASE(par_for_each_single)
{
    std::vector<int> v = {10};
    migraphx::par_for_each(v.begin(), v.end(), [](int& x) { x = x * x; });
    EXPECT(v == std::vector<int>{100});
}

TEST_CASE(par_for_each_exception)
{
    std::vector<int> v = {1, 2, 3};
    CHECK(test::throws([&] {
        migraphx::par_for_each(v.begin(), v.end(), [](int x) {
            if(x == 2)
                throw std::runtime_error("test error");
        });
    }));
}

TEST_CASE(par_for_each_large)
{
    std::vector<int> v(64);
    std::iota(v.begin(), v.end(), 0);
    std::vector<int> expected(v.size());
    std::transform(v.begin(), v.end(), expected.begin(), [](int x) { return x * x; });
    migraphx::par_for_each(v.begin(), v.end(), [](int& x) { x = x * x; });
    EXPECT(v == expected);
}

// par_copy_if tests

TEST_CASE(par_copy_if_basic)
{
    std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> output(input.size());
    auto it = migraphx::par_copy_if(
        input.begin(), input.end(), output.begin(), [](int x) { return x % 2 == 0; });
    EXPECT((it - output.begin()) == 4);
    EXPECT(*(it - 1) == 8);
    output.erase(it, output.end());
    EXPECT(output == std::vector<int>{2, 4, 6, 8});
}

TEST_CASE(par_copy_if_empty)
{
    std::vector<int> input;
    std::vector<int> output;
    auto it =
        migraphx::par_copy_if(input.begin(), input.end(), output.begin(), [](int) { return true; });
    EXPECT(it == output.begin());
    EXPECT(output.empty());
}

TEST_CASE(par_copy_if_none_match)
{
    std::vector<int> input = {1, 3, 5, 7};
    std::vector<int> output(input.size());
    auto it = migraphx::par_copy_if(
        input.begin(), input.end(), output.begin(), [](int x) { return x % 2 == 0; });
    EXPECT(it == output.begin());
    output.erase(it, output.end());
    EXPECT(output.empty());
}

TEST_CASE(par_copy_if_all_match)
{
    std::vector<int> input = {2, 4, 6};
    std::vector<int> output(input.size());
    auto it = migraphx::par_copy_if(
        input.begin(), input.end(), output.begin(), [](int x) { return x % 2 == 0; });
    EXPECT(it == output.end());
    EXPECT(*(it - 1) == 6);
    EXPECT(output == std::vector<int>{2, 4, 6});
}

TEST_CASE(par_copy_if_to_sized_output)
{
    std::vector<int> input = {10, 21, 30, 41, 50};
    std::vector<int> output(input.size());
    auto it = migraphx::par_copy_if(
        input.begin(), input.end(), output.begin(), [](int x) { return x % 10 == 0; });
    EXPECT((it - output.begin()) == 3);
    EXPECT(*(it - 1) == 50);
    output.erase(it, output.end());
    EXPECT(output == std::vector<int>{10, 30, 50});
}

// par_sort tests

TEST_CASE(par_sort_basic)
{
    std::vector<int> v = {5, 3, 1, 4, 2};
    migraphx::par_sort(v.begin(), v.end());
    EXPECT(v == std::vector<int>{1, 2, 3, 4, 5});
}

TEST_CASE(par_sort_empty)
{
    std::vector<int> v;
    migraphx::par_sort(v.begin(), v.end());
    EXPECT(v.empty());
}

TEST_CASE(par_sort_single)
{
    std::vector<int> v = {42};
    migraphx::par_sort(v.begin(), v.end());
    EXPECT(v == std::vector<int>{42});
}

TEST_CASE(par_sort_already_sorted)
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    migraphx::par_sort(v.begin(), v.end());
    EXPECT(v == std::vector<int>{1, 2, 3, 4, 5});
}

TEST_CASE(par_sort_reverse)
{
    std::vector<int> v = {5, 4, 3, 2, 1};
    migraphx::par_sort(v.begin(), v.end());
    EXPECT(v == std::vector<int>{1, 2, 3, 4, 5});
}

TEST_CASE(par_sort_duplicates)
{
    std::vector<int> v = {3, 1, 2, 1, 3, 2};
    migraphx::par_sort(v.begin(), v.end());
    EXPECT(v == std::vector<int>{1, 1, 2, 2, 3, 3});
}

TEST_CASE(par_sort_custom_comparator)
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    migraphx::par_sort(v.begin(), v.end(), std::greater<>{});
    EXPECT(v == std::vector<int>{5, 4, 3, 2, 1});
}

TEST_CASE(par_sort_large)
{
    std::vector<int> v(64);
    std::iota(v.begin(), v.end(), 0);
    std::vector<int> expected = v;
    std::reverse(v.begin(), v.end());
    migraphx::par_sort(v.begin(), v.end());
    EXPECT(v == expected);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
