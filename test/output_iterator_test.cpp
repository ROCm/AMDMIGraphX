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
#include <migraphx/output_iterator.hpp>
#include <algorithm>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include "test.hpp"

TEST_CASE(function_output_iterator_collect)
{
    std::vector<int> result;
    std::vector<int> input = {1, 2, 3};
    std::transform(
        input.begin(),
        input.end(),
        migraphx::make_function_output_iterator([&](const auto& x) { result.push_back(x); }),
        [](int x) { return x * 2; });
    EXPECT(result == std::vector<int>{2, 4, 6});
}

TEST_CASE(join_back_inserter_flatten)
{
    std::vector<std::vector<int>> input = {{1, 2}, {}, {3}};
    std::vector<int> result;
    std::copy(input.begin(), input.end(), migraphx::join_back_inserter(result));
    EXPECT(result == std::vector<int>{1, 2, 3});
}

TEST_CASE(function_output_iterator_adaptor_assign)
{
    std::vector<int> result = {0, 0, 0};
    std::vector<int> input  = {1, 2, 3};
    std::copy(input.begin(),
              input.end(),
              migraphx::make_function_output_iterator_adaptor(
                  result.begin(), [](int& x, int value) { x = value + 1; }));
    EXPECT(result == std::vector<int>{2, 3, 4});
}

TEST_CASE(element_output_iterator_map_values)
{
    std::map<std::string, int> m = {{"a", 1}, {"b", 2}, {"c", 3}};
    std::transform(m.begin(),
                   m.end(),
                   migraphx::element_output_iterator<1>(m.begin()),
                   [](auto&& p) { return p.second * 2; });
    std::map<std::string, int> expected = {{"a", 2}, {"b", 4}, {"c", 6}};
    EXPECT(m == expected);
}

TEST_CASE(element_output_iterator_pair_first)
{
    std::vector<std::pair<int, std::string>> v = {{1, "x"}, {2, "y"}};
    std::vector<int> keys                      = {10, 20};
    std::copy(keys.begin(), keys.end(), migraphx::element_output_iterator<0>(v.begin()));
    std::vector<std::pair<int, std::string>> expected = {{10, "x"}, {20, "y"}};
    EXPECT(v == expected);
}

TEST_CASE(element_output_iterator_tuple)
{
    std::vector<std::tuple<int, int, int>> v = {{1, 2, 3}, {4, 5, 6}};
    std::vector<int> input                   = {7, 8};
    std::copy(input.begin(), input.end(), migraphx::element_output_iterator<2>(v.begin()));
    std::vector<std::tuple<int, int, int>> expected = {{1, 2, 7}, {4, 5, 8}};
    EXPECT(v == expected);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
