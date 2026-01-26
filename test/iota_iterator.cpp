/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include <migraphx/iota_iterator.hpp>
#include <migraphx/functional.hpp>
#include <algorithm>
#include <array>
#include <numeric>
#include <vector>
#include <cstdint>
#include <limits>
#include <test.hpp>

TEST_CASE(iota_iterator_basic)
{
    // Test basic iota_iterator with identity function
    migraphx::iota_iterator begin{0};
    migraphx::iota_iterator end{5};

    // Check initial values
    EXPECT(*begin == 0);
    EXPECT(*(end - 5) == 0);

    // Check dereferencing
    EXPECT(*(begin + 1) == 1);
    EXPECT(*(end - 1) == 4);

    // Check distance
    EXPECT(end - begin == 5);

    // Collect values using the iterator
    std::vector<int> result;
    std::copy(begin, end, std::back_inserter(result));

    // Verify results
    std::vector<int> expected = {0, 1, 2, 3, 4};
    EXPECT(result == expected);
}

TEST_CASE(iota_iterator_custom_function)
{
    // Test with a custom function that doubles the index
    auto double_it = [](std::ptrdiff_t i) { return i * 2; };
    auto begin     = migraphx::make_basic_iota_iterator(0, double_it);
    auto end       = migraphx::make_basic_iota_iterator(5, double_it);

    // Check dereferencing
    EXPECT(*begin == 0);
    EXPECT(*(begin + 1) == 2);
    EXPECT(*(begin + 2) == 4);
    EXPECT(*(begin + 3) == 6);
    EXPECT(*(begin + 4) == 8);

    // Collect values using the iterator
    std::vector<int> result;
    std::copy(begin, end, std::back_inserter(result));

    // Verify results
    std::vector<int> expected = {0, 2, 4, 6, 8};
    EXPECT(result == expected);
}

TEST_CASE(iota_iterator_increment_decrement)
{
    migraphx::iota_iterator it{3};

    // Pre-increment
    auto& ref1 = ++it;
    EXPECT(*it == 4);
    EXPECT(&ref1 == &it); // Check that pre-increment returns reference

    // Post-increment
    auto it2 = it++;
    EXPECT(*it == 5);
    EXPECT(*it2 == 4);

    // Pre-decrement
    auto& ref2 = --it;
    EXPECT(*it == 4);
    EXPECT(&ref2 == &it); // Check that pre-decrement returns reference

    // Post-decrement
    auto it3 = it--;
    EXPECT(*it == 3);
    EXPECT(*it3 == 4);
}

TEST_CASE(iota_iterator_increment_decrement_temp)
{
    auto dec = --migraphx::iota_iterator{3};
    EXPECT(*dec == 2); // Test pre-decrement on temporary

    auto inc = ++migraphx::iota_iterator{3};
    EXPECT(*inc == 4); // Test pre-increment on temporary
}

TEST_CASE(iota_iterator_compound_assignment)
{
    migraphx::iota_iterator it{5};

    // += operator
    auto& ref1 = it += 3;
    EXPECT(*it == 8);
    EXPECT(&ref1 == &it); // Check that += returns reference

    // -= operator
    auto& ref2 = it -= 5;
    EXPECT(*it == 3);
    EXPECT(&ref2 == &it); // Check that -= returns reference
}

TEST_CASE(iota_iterator_arithmetic)
{
    migraphx::iota_iterator it{5};

    // Addition (it + n)
    auto it1 = it + 3;
    EXPECT(*it1 == 8);
    EXPECT(*it == 5); // Original unchanged

    // Addition (n + it)
    auto it2 = 2 + it;
    EXPECT(*it2 == 7);
    EXPECT(*it == 5); // Original unchanged

    // Subtraction (it - n)
    auto it3 = it - 2;
    EXPECT(*it3 == 3);
    EXPECT(*it == 5); // Original unchanged

    // Difference (it1 - it2)
    EXPECT(it1 - it2 == 1);
    EXPECT(it2 - it1 == -1);
    EXPECT(it1 - it == 3);
}

TEST_CASE(iota_iterator_comparisons)
{
    migraphx::iota_iterator it1{5};
    migraphx::iota_iterator it2{5};
    migraphx::iota_iterator it3{8};

    // Equality
    EXPECT(it1 == it2);
    EXPECT(not(it1 == it3));

    // Inequality
    EXPECT(it1 != it3);
    EXPECT(not(it1 != it2));

    // Less than
    EXPECT(it1 < it3);
    EXPECT(not(it3 < it1));
    EXPECT(not(it1 < it2));

    // Greater than
    EXPECT(it3 > it1);
    EXPECT(not(it1 > it3));
    EXPECT(not(it1 > it2));

    // Less than or equal
    EXPECT(it1 <= it3);
    EXPECT(it1 <= it2);
    EXPECT(not(it3 <= it1));

    // Greater than or equal
    EXPECT(it3 >= it1);
    EXPECT(it1 >= it2);
    EXPECT(not(it1 >= it3));
}

TEST_CASE(iota_iterator_indexing)
{
    migraphx::iota_iterator it{10};

    // Test subscript operator
    EXPECT(it[0] == 10);
    EXPECT(it[1] == 11);
    EXPECT(it[5] == 15);
    EXPECT(it[-2] == 8);
}

TEST_CASE(iota_iterator_indexing_custom_function)
{
    std::array<int, 20> data;
    std::iota(data.begin(), data.end(), 0);
    auto it = migraphx::make_basic_iota_iterator(10, [&](auto i) -> const int& { return data[i]; });

    // Test subscript operator
    EXPECT(it[0] == 10);
    EXPECT(it[1] == 11);
    EXPECT(it[5] == 15);
    EXPECT(it[-2] == 8);
}

TEST_CASE(iota_iterator_custom_type)
{
    // Using a custom struct as the function result
    struct point
    {
        int x;
        int y;
    };

    auto point_maker = [](std::ptrdiff_t i) -> point {
        return {static_cast<int>(i), static_cast<int>(i * i)};
    };

    auto begin = migraphx::make_basic_iota_iterator(0, point_maker);

    // Test dereferencing and arrow operator
    EXPECT((*begin).x == 0);
    EXPECT((*begin).y == 0);
    EXPECT(begin->x == 0);
    EXPECT(begin->y == 0);

    // Advance and check
    auto it = begin + 3;
    EXPECT(it->x == 3);
    EXPECT(it->y == 9);

    // Check subscript
    EXPECT(begin[4].x == 4);
    EXPECT(begin[4].y == 16);
}

TEST_CASE(iota_iterator_algorithm_compatibility)
{
    // Test that iota_iterator works with standard algorithms
    migraphx::iota_iterator begin{0};
    migraphx::iota_iterator end{10};

    // Test with std::accumulate
    auto sum = std::accumulate(begin, end, std::ptrdiff_t{0});
    EXPECT(sum == 45); // Sum of 0-9 is 45

    // Test with std::transform
    std::vector<int> result(10);
    std::transform(begin, end, result.begin(), [](int i) { return i * i; });

    std::vector<int> expected = {0, 1, 4, 9, 16, 25, 36, 49, 64, 81};
    EXPECT(result == expected);

    // Test with std::find
    auto found = std::find_if(begin, end, [](int i) { return i > 5; });
    EXPECT(*found == 6);
    EXPECT(found - begin == 6);

    // Test with std::sort (using a vector of iota_iterators)
    std::vector<migraphx::iota_iterator> iters;
    for(int i = 9; i >= 0; --i)
        iters.push_back(migraphx::iota_iterator{i});

    std::sort(iters.begin(), iters.end());

    for(size_t i = 0; i < iters.size(); ++i)
        EXPECT(*iters[i] == static_cast<std::ptrdiff_t>(i));
}

TEST_CASE(iota_iterator_large_values)
{
    // Define values larger than uint32_t max
    const std::ptrdiff_t uint32_max = std::numeric_limits<uint32_t>::max();
    const std::ptrdiff_t base_val   = uint32_max + 10;
    const std::ptrdiff_t offset1    = 500;
    const std::ptrdiff_t offset2    = 1000;

    // Test with + operator (it + n)
    {
        migraphx::iota_iterator it{base_val};
        auto it_plus = it + offset1;
        EXPECT(*it_plus == base_val + offset1);
        EXPECT(*it == base_val); // Original unchanged
    }

    // Test with + operator (n + it)
    {
        migraphx::iota_iterator it{base_val};
        auto it_plus = offset1 + it;
        EXPECT(*it_plus == base_val + offset1);
        EXPECT(*it == base_val); // Original unchanged
    }

    // Test with - operator
    {
        migraphx::iota_iterator it{base_val + offset2};
        auto it_minus = it - offset1;
        EXPECT(*it_minus == base_val + offset2 - offset1);
        EXPECT(*it == base_val + offset2); // Original unchanged
    }

    // Test with += operator
    {
        migraphx::iota_iterator it{base_val};
        it += offset2;
        EXPECT(*it == base_val + offset2);
    }

    // Test with -= operator
    {
        migraphx::iota_iterator it{base_val + offset2};
        it -= offset1;
        EXPECT(*it == base_val + offset2 - offset1);
    }

    // Test with [] operator
    {
        migraphx::iota_iterator it{base_val};
        EXPECT(it[0] == base_val);
        EXPECT(it[offset1] == base_val + offset1);
        EXPECT(it[-offset1] == base_val - offset1);
    }

    // Test iterator difference
    {
        migraphx::iota_iterator it1{base_val};
        migraphx::iota_iterator it2{base_val + offset2};
        EXPECT(it2 - it1 == offset2);
        EXPECT(it1 - it2 == -offset2);
    }

    // Test with larger than int32_t offsets
    const std::ptrdiff_t large_offset = static_cast<std::ptrdiff_t>(3) * uint32_max;

    // Test with + operator
    migraphx::iota_iterator it_large{1};
    auto it_large_plus = it_large + large_offset;
    EXPECT(*it_large_plus == 1 + large_offset);

    // Test with += operator
    migraphx::iota_iterator it_large2{1};
    it_large2 += large_offset;
    EXPECT(*it_large2 == 1 + large_offset);

    // Test with [] operator
    EXPECT(it_large[large_offset] == 1 + large_offset);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
