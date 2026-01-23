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
#include <migraphx/kernels/algorithm.hpp>
#include <migraphx/kernels/array.hpp>
#include <migraphx/kernels/float_equal.hpp>
#include <migraphx/kernels/test.hpp>

struct empty_range
{
    constexpr int* begin() { return nullptr; }

    constexpr int* end() { return nullptr; }
};

TEST_CASE(swap_basic)
{
    int a = 5;
    int b = 10;
    migraphx::swap(a, b);
    EXPECT(a == 10 and b == 5);
}
TEST_CASE(swap_same_value)
{
    int a = 7;
    int b = 7;
    migraphx::swap(a, b);
    EXPECT(a == 7 and b == 7);
}
TEST_CASE(swap_different_types)
{
    double a = 3.14;
    double b = 2.71;
    migraphx::swap(a, b);
    EXPECT(migraphx::float_equal(a, 2.71) and migraphx::float_equal(b, 3.14));
}
TEST_CASE(iter_swap_basic)
{
    migraphx::array<int, 3> arr = {1, 2, 3};
    migraphx::iter_swap(arr.begin(), arr.begin() + 2);
    migraphx::array<int, 3> expected = {3, 2, 1};
    EXPECT(arr == expected);
}
TEST_CASE(iter_swap_same_iterator)
{
    migraphx::array<int, 3> arr      = {1, 2, 3};
    migraphx::array<int, 3> original = arr;
    migraphx::iter_swap(arr.begin(), arr.begin());
    EXPECT(arr == original);
}
TEST_CASE(iter_swap_adjacent)
{
    migraphx::array<int, 4> arr = {10, 20, 30, 40};
    migraphx::iter_swap(arr.begin() + 1, arr.begin() + 2);
    migraphx::array<int, 4> expected = {10, 30, 20, 40};
    EXPECT(arr == expected);
}

TEST_CASE(less_functor)
{
    migraphx::less comp;
    EXPECT(comp(1, 2));
    EXPECT(not comp(2, 1));
    EXPECT(not comp(5, 5));
}
TEST_CASE(greater_functor)
{
    migraphx::greater comp;
    EXPECT(comp(2, 1));
    EXPECT(not comp(1, 2));
    EXPECT(not comp(5, 5));
}

TEST_CASE(accumulate_basic)
{
    migraphx::array<int, 4> arr = {1, 2, 3, 4};
    auto sum = migraphx::accumulate(arr.begin(), arr.end(), 0, [](int a, int b) { return a + b; });
    EXPECT(sum == 10);
}
TEST_CASE(accumulate_empty_range)
{
    empty_range arr = {};
    auto sum =
        migraphx::accumulate(arr.begin(), arr.begin(), 42, [](int a, int b) { return a + b; });
    EXPECT(sum == 42);
}
TEST_CASE(accumulate_multiply)
{
    migraphx::array<int, 3> arr = {2, 3, 4};
    auto product =
        migraphx::accumulate(arr.begin(), arr.end(), 1, [](int a, int b) { return a * b; });
    EXPECT(product == 24);
}
TEST_CASE(accumulate_single_element)
{
    migraphx::array<int, 1> arr = {7};
    auto sum = migraphx::accumulate(arr.begin(), arr.end(), 5, [](int a, int b) { return a + b; });
    EXPECT(sum == 12);
}

TEST_CASE(copy_basic)
{
    migraphx::array<int, 4> src = {1, 2, 3, 4};
    migraphx::array<int, 4> dst = {0, 0, 0, 0};
    migraphx::copy(src.begin(), src.end(), dst.begin());
    EXPECT(src == dst);
}
TEST_CASE(copy_empty_range)
{
    migraphx::array<int, 3> src = {1, 2, 3};
    migraphx::array<int, 3> dst = {0, 0, 0};
    auto* result                = migraphx::copy(src.begin(), src.begin(), dst.begin());
    EXPECT(result == dst.begin());
    migraphx::array<int, 3> expected = {0, 0, 0};
    EXPECT(dst == expected);
}
TEST_CASE(copy_partial)
{
    migraphx::array<int, 5> src = {10, 20, 30, 40, 50};
    migraphx::array<int, 3> dst = {0, 0, 0};
    migraphx::copy(src.begin() + 1, src.begin() + 4, dst.begin());
    migraphx::array<int, 3> expected = {20, 30, 40};
    EXPECT(dst == expected);
}
TEST_CASE(copy_if_basic)
{
    migraphx::array<int, 6> src = {1, 2, 3, 4, 5, 6};
    migraphx::array<int, 6> dst = {0, 0, 0, 0, 0, 0};
    auto* end_it =
        migraphx::copy_if(src.begin(), src.end(), dst.begin(), [](int x) { return x % 2 == 0; });
    EXPECT(dst[0] == 2);
    EXPECT(dst[1] == 4);
    EXPECT(dst[2] == 6);
    EXPECT(end_it == dst.begin() + 3);
}
TEST_CASE(copy_if_none_match)
{
    migraphx::array<int, 3> src = {1, 3, 5};
    migraphx::array<int, 3> dst = {0, 0, 0};
    auto* end_it =
        migraphx::copy_if(src.begin(), src.end(), dst.begin(), [](int x) { return x % 2 == 0; });
    EXPECT(end_it == dst.begin());
    migraphx::array<int, 3> expected = {0, 0, 0};
    EXPECT(dst == expected);
}
TEST_CASE(copy_if_all_match)
{
    migraphx::array<int, 3> src = {2, 4, 6};
    migraphx::array<int, 3> dst = {0, 0, 0};
    auto* end_it =
        migraphx::copy_if(src.begin(), src.end(), dst.begin(), [](int x) { return x % 2 == 0; });
    EXPECT(end_it == dst.end());
    EXPECT(src == dst);
}

TEST_CASE(count_if_basic)
{
    migraphx::array<int, 6> arr = {1, 2, 3, 4, 5, 6};
    auto count = migraphx::count_if(arr.begin(), arr.end(), [](int x) { return x % 2 == 0; });
    EXPECT(count == 3);
}
TEST_CASE(count_if_empty_range)
{
    empty_range arr = {};
    auto count      = migraphx::count_if(arr.begin(), arr.end(), [](int x) { return x > 0; });
    EXPECT(count == 0);
}
TEST_CASE(count_if_none_match)
{
    migraphx::array<int, 4> arr = {1, 3, 5, 7};
    auto count = migraphx::count_if(arr.begin(), arr.end(), [](int x) { return x % 2 == 0; });
    EXPECT(count == 0);
}
TEST_CASE(count_if_all_match)
{
    migraphx::array<int, 4> arr = {2, 4, 6, 8};
    auto count = migraphx::count_if(arr.begin(), arr.end(), [](int x) { return x % 2 == 0; });
    EXPECT(count == 4);
}
TEST_CASE(count_if_single_element_match)
{
    migraphx::array<int, 1> arr = {4};
    auto count = migraphx::count_if(arr.begin(), arr.end(), [](int x) { return x % 2 == 0; });
    EXPECT(count == 1);
}
TEST_CASE(count_if_single_element_no_match)
{
    migraphx::array<int, 1> arr = {3};
    auto count = migraphx::count_if(arr.begin(), arr.end(), [](int x) { return x % 2 == 0; });
    EXPECT(count == 0);
}
TEST_CASE(count_if_greater_than)
{
    migraphx::array<int, 5> arr = {1, 5, 10, 15, 20};
    auto count = migraphx::count_if(arr.begin(), arr.end(), [](int x) { return x > 7; });
    EXPECT(count == 3);
}
TEST_CASE(count_if_partial_range)
{
    migraphx::array<int, 8> arr = {1, 2, 3, 4, 5, 6, 7, 8};
    auto count =
        migraphx::count_if(arr.begin() + 2, arr.begin() + 6, [](int x) { return x % 2 == 0; });
    EXPECT(count == 2);
}

TEST_CASE(is_sorted_until_sorted)
{
    migraphx::array<int, 4> arr = {1, 2, 3, 4};
    auto* result = migraphx::is_sorted_until(arr.begin(), arr.end(), migraphx::less{});
    EXPECT(result == arr.end());
}
TEST_CASE(is_sorted_until_unsorted)
{
    migraphx::array<int, 5> arr = {1, 2, 4, 3, 5};
    auto* result = migraphx::is_sorted_until(arr.begin(), arr.end(), migraphx::less{});
    EXPECT(result == arr.begin() + 3);
}
TEST_CASE(is_sorted_until_empty)
{
    empty_range arr = {};
    auto* result    = migraphx::is_sorted_until(arr.begin(), arr.end(), migraphx::less{});
    EXPECT(result == arr.end());
}
TEST_CASE(is_sorted_until_single)
{
    migraphx::array<int, 1> arr = {42};
    auto* result = migraphx::is_sorted_until(arr.begin(), arr.end(), migraphx::less{});
    EXPECT(result == arr.end());
}
TEST_CASE(is_sorted_until_descending)
{
    migraphx::array<int, 4> arr = {4, 3, 2, 1};
    auto* result = migraphx::is_sorted_until(arr.begin(), arr.end(), migraphx::greater{});
    EXPECT(result == arr.end());
}
TEST_CASE(is_sorted_true)
{
    migraphx::array<int, 4> arr = {1, 2, 3, 4};
    EXPECT(migraphx::is_sorted(arr.begin(), arr.end(), migraphx::less{}));
}
TEST_CASE(is_sorted_false)
{
    migraphx::array<int, 4> arr = {1, 3, 2, 4};
    EXPECT(not migraphx::is_sorted(arr.begin(), arr.end(), migraphx::less{}));
}
TEST_CASE(is_sorted_empty)
{
    empty_range arr = {};
    EXPECT(migraphx::is_sorted(arr.begin(), arr.end(), migraphx::less{}));
}
TEST_CASE(is_sorted_duplicates)
{
    migraphx::array<int, 4> arr = {1, 2, 2, 3};
    EXPECT(migraphx::is_sorted(arr.begin(), arr.end(), migraphx::less{}));
}

TEST_CASE(for_each_basic)
{
    migraphx::array<int, 3> arr = {1, 2, 3};
    int sum                     = 0;
    migraphx::for_each(arr.begin(), arr.end(), [&sum](int x) { sum += x; });
    EXPECT(sum == 6);
}
TEST_CASE(for_each_modify)
{
    migraphx::array<int, 3> arr = {1, 2, 3};
    migraphx::for_each(arr.begin(), arr.end(), [](int& x) { x *= 2; });
    migraphx::array<int, 3> expected = {2, 4, 6};
    EXPECT(arr == expected);
}
TEST_CASE(for_each_empty)
{
    empty_range arr = {};
    int count       = 0;
    migraphx::for_each(arr.begin(), arr.end(), [&count](int) { count++; });
    EXPECT(count == 0);
}

TEST_CASE(find_if_found)
{
    migraphx::array<int, 5> arr = {1, 3, 5, 7, 9};
    auto* result = migraphx::find_if(arr.begin(), arr.end(), [](int x) { return x > 5; });
    EXPECT(result == arr.begin() + 3);
}
TEST_CASE(find_if_not_found)
{
    migraphx::array<int, 4> arr = {1, 3, 5, 7};
    auto* result = migraphx::find_if(arr.begin(), arr.end(), [](int x) { return x > 10; });
    EXPECT(result == arr.end());
}
TEST_CASE(find_if_first_element)
{
    migraphx::array<int, 4> arr = {10, 3, 5, 7};
    auto* result = migraphx::find_if(arr.begin(), arr.end(), [](int x) { return x > 5; });
    EXPECT(result == arr.begin());
}
TEST_CASE(find_if_empty)
{
    empty_range arr = {};
    auto* result    = migraphx::find_if(arr.begin(), arr.end(), [](int x) { return x > 0; });
    EXPECT(result == arr.end());
}
TEST_CASE(find_basic)
{
    migraphx::array<int, 4> arr = {10, 20, 30, 40};
    auto* result                = migraphx::find(arr.begin(), arr.end(), 30);
    EXPECT(result == arr.begin() + 2);
}
TEST_CASE(find_not_found)
{
    migraphx::array<int, 4> arr = {10, 20, 30, 40};
    auto* result                = migraphx::find(arr.begin(), arr.end(), 50);
    EXPECT(result == arr.end());
}
TEST_CASE(find_first_element)
{
    migraphx::array<int, 4> arr = {10, 20, 30, 40};
    auto* result                = migraphx::find(arr.begin(), arr.end(), 10);
    EXPECT(result == arr.begin());
}
TEST_CASE(find_duplicates)
{
    migraphx::array<int, 5> arr = {10, 20, 30, 20, 40};
    auto* result                = migraphx::find(arr.begin(), arr.end(), 20);
    EXPECT(result == arr.begin() + 1);
}

TEST_CASE(any_of_true)
{
    migraphx::array<int, 4> arr = {1, 3, 5, 7};
    EXPECT(migraphx::any_of(arr.begin(), arr.end(), [](int x) { return x > 5; }));
}
TEST_CASE(any_of_false)
{
    migraphx::array<int, 4> arr = {1, 3, 5, 7};
    EXPECT(not migraphx::any_of(arr.begin(), arr.end(), [](int x) { return x > 10; }));
}
TEST_CASE(any_of_empty)
{
    empty_range arr = {};
    EXPECT(not migraphx::any_of(arr.begin(), arr.end(), [](int x) { return x > 0; }));
}
TEST_CASE(any_of_first_element)
{
    migraphx::array<int, 4> arr = {10, 1, 2, 3};
    EXPECT(migraphx::any_of(arr.begin(), arr.end(), [](int x) { return x > 5; }));
}
TEST_CASE(none_of_true)
{
    migraphx::array<int, 4> arr = {1, 3, 5, 7};
    EXPECT(migraphx::none_of(arr.begin(), arr.end(), [](int x) { return x > 10; }));
}
TEST_CASE(none_of_false)
{
    migraphx::array<int, 4> arr = {1, 3, 5, 7};
    EXPECT(not migraphx::none_of(arr.begin(), arr.end(), [](int x) { return x > 5; }));
}
TEST_CASE(none_of_empty)
{
    empty_range arr = {};
    EXPECT(migraphx::none_of(arr.begin(), arr.end(), [](int x) { return x > 0; }));
}
TEST_CASE(all_of_true)
{
    migraphx::array<int, 4> arr = {2, 4, 6, 8};
    EXPECT(migraphx::all_of(arr.begin(), arr.end(), [](int x) { return x % 2 == 0; }));
}
TEST_CASE(all_of_false)
{
    migraphx::array<int, 4> arr = {2, 4, 5, 8};
    EXPECT(not migraphx::all_of(arr.begin(), arr.end(), [](int x) { return x % 2 == 0; }));
}
TEST_CASE(all_of_empty)
{
    empty_range arr = {};
    EXPECT(migraphx::all_of(arr.begin(), arr.end(), [](int x) { return x > 0; }));
}
TEST_CASE(all_of_single_true)
{
    migraphx::array<int, 1> arr = {4};
    EXPECT(migraphx::all_of(arr.begin(), arr.end(), [](int x) { return x % 2 == 0; }));
}
TEST_CASE(all_of_single_false)
{
    migraphx::array<int, 1> arr = {5};
    EXPECT(not migraphx::all_of(arr.begin(), arr.end(), [](int x) { return x % 2 == 0; }));
}

TEST_CASE(search_found)
{
    migraphx::array<int, 6> haystack = {1, 2, 3, 4, 3, 4};
    migraphx::array<int, 2> needle   = {3, 4};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 2);
}
TEST_CASE(search_not_found)
{
    migraphx::array<int, 5> haystack = {1, 2, 3, 4, 5};
    migraphx::array<int, 2> needle   = {6, 7};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.end());
}
TEST_CASE(search_empty_needle)
{
    migraphx::array<int, 3> haystack = {1, 2, 3};
    empty_range needle               = {};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin());
}
TEST_CASE(search_empty_haystack)
{
    empty_range haystack           = {};
    migraphx::array<int, 1> needle = {1};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.end());
}
TEST_CASE(search_exact_match)
{
    migraphx::array<int, 3> haystack = {1, 2, 3};
    migraphx::array<int, 3> needle   = {1, 2, 3};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin());
}
TEST_CASE(search_partial_match)
{
    migraphx::array<int, 5> haystack = {1, 2, 1, 2, 3};
    migraphx::array<int, 3> needle   = {1, 2, 3};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 2);
}
// Additional search tests (extensive)
TEST_CASE(search_overlapping_pattern)
{
    migraphx::array<int, 8> haystack = {1, 2, 1, 2, 1, 2, 3, 4};
    migraphx::array<int, 3> needle   = {1, 2, 3};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 4);
}
TEST_CASE(search_pattern_at_end)
{
    migraphx::array<int, 6> haystack = {1, 2, 3, 4, 5, 6};
    migraphx::array<int, 2> needle   = {5, 6};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 4);
}
TEST_CASE(search_pattern_at_beginning)
{
    migraphx::array<int, 6> haystack = {1, 2, 3, 4, 5, 6};
    migraphx::array<int, 2> needle   = {1, 2};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin());
}
TEST_CASE(search_single_element_pattern)
{
    migraphx::array<int, 5> haystack = {1, 3, 5, 7, 9};
    migraphx::array<int, 1> needle   = {5};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 2);
}
TEST_CASE(search_single_element_pattern_not_found)
{
    migraphx::array<int, 5> haystack = {1, 3, 5, 7, 9};
    migraphx::array<int, 1> needle   = {4};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.end());
}
TEST_CASE(search_repeated_elements)
{
    migraphx::array<int, 8> haystack = {2, 2, 2, 2, 1, 2, 2, 3};
    migraphx::array<int, 3> needle   = {2, 2, 3};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 5);
}
TEST_CASE(search_partial_match_backtrack)
{
    migraphx::array<int, 10> haystack = {1, 2, 3, 1, 2, 4, 1, 2, 3, 4};
    migraphx::array<int, 4> needle    = {1, 2, 3, 4};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 6);
}
TEST_CASE(search_multiple_false_starts)
{
    migraphx::array<int, 12> haystack = {1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 5, 6};
    migraphx::array<int, 4> needle    = {1, 2, 3, 4};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 6);
}
TEST_CASE(search_needle_longer_than_haystack)
{
    migraphx::array<int, 3> haystack = {1, 2, 3};
    migraphx::array<int, 5> needle   = {1, 2, 3, 4, 5};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.end());
}
TEST_CASE(search_identical_arrays)
{
    migraphx::array<int, 4> haystack = {5, 10, 15, 20};
    migraphx::array<int, 4> needle   = {5, 10, 15, 20};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin());
}
TEST_CASE(search_all_same_elements)
{
    migraphx::array<int, 6> haystack = {3, 3, 3, 3, 3, 3};
    migraphx::array<int, 3> needle   = {3, 3, 3};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin());
}
TEST_CASE(search_pattern_appears_multiple_times)
{
    migraphx::array<int, 9> haystack = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    migraphx::array<int, 2> needle   = {2, 3};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 1);
}
TEST_CASE(search_partial_range)
{
    migraphx::array<int, 10> haystack = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    migraphx::array<int, 2> needle    = {4, 5};
    auto* result =
        migraphx::search(haystack.begin() + 2, haystack.begin() + 7, needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 4);
}
TEST_CASE(search_stress_long_pattern)
{
    migraphx::array<int, 15> haystack = {1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1};
    migraphx::array<int, 6> needle    = {2, 3, 4, 5, 6, 7};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 5);
}
TEST_CASE(search_stress_worst_case_complexity)
{
    migraphx::array<int, 8> haystack = {1, 1, 1, 1, 1, 1, 1, 2};
    migraphx::array<int, 7> needle   = {1, 1, 1, 1, 1, 1, 2};
    auto* result = migraphx::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 1);
}

TEST_CASE(inner_product_basic)
{
    migraphx::array<int, 3> a = {1, 2, 3};
    migraphx::array<int, 3> b = {4, 5, 6};
    auto result               = migraphx::inner_product(a.begin(), a.end(), b.begin(), 0);
    EXPECT(result == 32);
}
TEST_CASE(inner_product_custom_ops)
{
    migraphx::array<int, 3> a = {1, 2, 3};
    migraphx::array<int, 3> b = {4, 5, 6};
    auto result               = migraphx::inner_product(
        a.begin(),
        a.end(),
        b.begin(),
        1,
        [](int x, int y) { return x * y; },
        [](int x, int y) { return x + y; });
    EXPECT(result == 315);
}
TEST_CASE(inner_product_empty)
{
    empty_range a = {};
    empty_range b = {};
    auto result   = migraphx::inner_product(a.begin(), a.end(), b.begin(), 42);
    EXPECT(result == 42);
}
TEST_CASE(inner_product_single)
{
    migraphx::array<int, 1> a = {3};
    migraphx::array<int, 1> b = {7};
    auto result               = migraphx::inner_product(a.begin(), a.end(), b.begin(), 5);
    EXPECT(result == 26);
}

TEST_CASE(equal_true)
{
    migraphx::array<int, 4> a = {1, 2, 3, 4};
    migraphx::array<int, 4> b = {1, 2, 3, 4};
    EXPECT(migraphx::equal(a.begin(), a.end(), b.begin(), [](int x, int y) { return x == y; }));
}
TEST_CASE(equal_false)
{
    migraphx::array<int, 4> a = {1, 2, 3, 4};
    migraphx::array<int, 4> b = {1, 2, 3, 5};
    EXPECT(not migraphx::equal(a.begin(), a.end(), b.begin(), [](int x, int y) { return x == y; }));
}
TEST_CASE(equal_empty)
{
    empty_range a = {};
    empty_range b = {};
    EXPECT(migraphx::equal(a.begin(), a.end(), b.begin(), [](int x, int y) { return x == y; }));
}
TEST_CASE(equal_custom_predicate)
{
    migraphx::array<int, 3> a = {1, 2, 3};
    migraphx::array<int, 3> b = {2, 4, 6};
    EXPECT(migraphx::equal(a.begin(), a.end(), b.begin(), [](int x, int y) { return x * 2 == y; }));
}

TEST_CASE(iota_basic)
{
    migraphx::array<int, 5> arr = {0, 0, 0, 0, 0};
    migraphx::iota(arr.begin(), arr.end(), 10);
    migraphx::array<int, 5> expected = {10, 11, 12, 13, 14};
    EXPECT(arr == expected);
}
TEST_CASE(iota_empty)
{
    empty_range arr = {};
    migraphx::iota(arr.begin(), arr.end(), 5);
}
TEST_CASE(iota_single)
{
    migraphx::array<int, 1> arr = {0};
    migraphx::iota(arr.begin(), arr.end(), 42);
    EXPECT(arr[0] == 42);
}
TEST_CASE(iota_negative)
{
    migraphx::array<int, 3> arr = {0, 0, 0};
    migraphx::iota(arr.begin(), arr.end(), -5);
    migraphx::array<int, 3> expected = {-5, -4, -3};
    EXPECT(arr == expected);
}

TEST_CASE(min_element_basic)
{
    migraphx::array<int, 5> arr = {3, 1, 4, 1, 5};
    auto* result                = migraphx::min_element(arr.begin(), arr.end(), migraphx::less{});
    EXPECT(result == arr.begin() + 1);
}
TEST_CASE(min_element_empty)
{
    empty_range arr = {};
    auto* result    = migraphx::min_element(arr.begin(), arr.end(), migraphx::less{});
    EXPECT(result == arr.end());
}
TEST_CASE(min_element_single)
{
    migraphx::array<int, 1> arr = {42};
    auto* result                = migraphx::min_element(arr.begin(), arr.end(), migraphx::less{});
    EXPECT(result == arr.begin());
}
TEST_CASE(min_element_all_equal)
{
    migraphx::array<int, 4> arr = {5, 5, 5, 5};
    auto* result                = migraphx::min_element(arr.begin(), arr.end(), migraphx::less{});
    EXPECT(result == arr.begin());
}
TEST_CASE(min_element_custom_compare)
{
    migraphx::array<int, 4> arr = {1, 2, 3, 4};
    auto* result = migraphx::min_element(arr.begin(), arr.end(), migraphx::greater{});
    EXPECT(result == arr.begin() + 3);
}

TEST_CASE(rotate_left_by_two)
{
    migraphx::array<int, 6> arr      = {1, 2, 3, 4, 5, 6};
    auto* result                     = migraphx::rotate(arr.begin(), arr.begin() + 2, arr.end());
    migraphx::array<int, 6> expected = {3, 4, 5, 6, 1, 2};
    EXPECT(arr == expected);
    EXPECT(result == arr.begin() + 4);
}
TEST_CASE(rotate_right_by_one)
{
    migraphx::array<int, 5> arr      = {1, 2, 3, 4, 5};
    auto* result                     = migraphx::rotate(arr.begin(), arr.end() - 1, arr.end());
    migraphx::array<int, 5> expected = {5, 1, 2, 3, 4};
    EXPECT(arr == expected);
    EXPECT(result == arr.begin() + 1);
}
TEST_CASE(rotate_half_array)
{
    migraphx::array<int, 8> arr      = {1, 2, 3, 4, 5, 6, 7, 8};
    auto* result                     = migraphx::rotate(arr.begin(), arr.begin() + 4, arr.end());
    migraphx::array<int, 8> expected = {5, 6, 7, 8, 1, 2, 3, 4};
    EXPECT(arr == expected);
    EXPECT(result == arr.begin() + 4);
}
TEST_CASE(rotate_almost_full)
{
    migraphx::array<int, 5> arr      = {1, 2, 3, 4, 5};
    auto* result                     = migraphx::rotate(arr.begin(), arr.begin() + 4, arr.end());
    migraphx::array<int, 5> expected = {5, 1, 2, 3, 4};
    EXPECT(arr == expected);
    EXPECT(result == arr.begin() + 1);
}
TEST_CASE(rotate_two_elements)
{
    migraphx::array<int, 2> arr      = {1, 2};
    auto* result                     = migraphx::rotate(arr.begin(), arr.begin() + 1, arr.end());
    migraphx::array<int, 2> expected = {2, 1};
    EXPECT(arr == expected);
    EXPECT(result == arr.begin() + 1);
}
TEST_CASE(rotate_partial_range)
{
    migraphx::array<int, 7> arr = {1, 2, 3, 4, 5, 6, 7};
    auto* result = migraphx::rotate(arr.begin() + 1, arr.begin() + 3, arr.begin() + 5);
    migraphx::array<int, 7> expected = {1, 4, 5, 2, 3, 6, 7};
    EXPECT(arr == expected);
    EXPECT(result == arr.begin() + 3);
}
TEST_CASE(rotate_with_duplicates)
{
    migraphx::array<int, 6> arr      = {1, 1, 2, 2, 3, 3};
    auto* result                     = migraphx::rotate(arr.begin(), arr.begin() + 2, arr.end());
    migraphx::array<int, 6> expected = {2, 2, 3, 3, 1, 1};
    EXPECT(arr == expected);
    EXPECT(result == arr.begin() + 4);
}
TEST_CASE(rotate_stress_test_large_shift)
{
    migraphx::array<int, 10> arr      = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto* result                      = migraphx::rotate(arr.begin(), arr.begin() + 7, arr.end());
    migraphx::array<int, 10> expected = {7, 8, 9, 0, 1, 2, 3, 4, 5, 6};
    EXPECT(arr == expected);
    EXPECT(result == arr.begin() + 3);
}
TEST_CASE(rotate_edge_case_middle_equals_first)
{
    migraphx::array<int, 4> arr      = {1, 2, 3, 4};
    migraphx::array<int, 4> original = arr;
    auto* result                     = migraphx::rotate(arr.begin(), arr.begin(), arr.end());
    EXPECT(arr == original);
    EXPECT(result == arr.end());
}
TEST_CASE(rotate_edge_case_middle_equals_last)
{
    migraphx::array<int, 4> arr      = {1, 2, 3, 4};
    migraphx::array<int, 4> original = arr;
    auto* result                     = migraphx::rotate(arr.begin(), arr.end(), arr.end());
    EXPECT(arr == original);
    EXPECT(result == arr.begin());
}

TEST_CASE(upper_bound_basic)
{
    migraphx::array<int, 5> arr = {1, 2, 2, 3, 4};
    auto* result = migraphx::upper_bound(arr.begin(), arr.end(), 2, migraphx::less{});
    EXPECT(result == arr.begin() + 3);
}
TEST_CASE(upper_bound_not_found)
{
    migraphx::array<int, 4> arr = {1, 2, 3, 4};
    auto* result = migraphx::upper_bound(arr.begin(), arr.end(), 5, migraphx::less{});
    EXPECT(result == arr.end());
}
TEST_CASE(upper_bound_first_element)
{
    migraphx::array<int, 4> arr = {1, 2, 3, 4};
    auto* result = migraphx::upper_bound(arr.begin(), arr.end(), 0, migraphx::less{});
    EXPECT(result == arr.begin());
}
TEST_CASE(upper_bound_empty)
{
    empty_range arr = {};
    auto* result    = migraphx::upper_bound(arr.begin(), arr.end(), 5, migraphx::less{});
    EXPECT(result == arr.end());
}
TEST_CASE(upper_bound_all_equal)
{
    migraphx::array<int, 3> arr = {2, 2, 2};
    auto* result = migraphx::upper_bound(arr.begin(), arr.end(), 2, migraphx::less{});
    EXPECT(result == arr.end());
}
TEST_CASE(upper_bound_multiple_duplicates)
{
    migraphx::array<int, 8> arr = {1, 2, 2, 2, 2, 3, 4, 5};
    auto* result = migraphx::upper_bound(arr.begin(), arr.end(), 2, migraphx::less{});
    EXPECT(result == arr.begin() + 5);
}
TEST_CASE(upper_bound_all_smaller)
{
    migraphx::array<int, 5> arr = {1, 2, 3, 4, 5};
    auto* result = migraphx::upper_bound(arr.begin(), arr.end(), 0, migraphx::less{});
    EXPECT(result == arr.begin());
}
TEST_CASE(upper_bound_all_larger)
{
    migraphx::array<int, 5> arr = {1, 2, 3, 4, 5};
    auto* result = migraphx::upper_bound(arr.begin(), arr.end(), 10, migraphx::less{});
    EXPECT(result == arr.end());
}
TEST_CASE(upper_bound_single_element_match)
{
    migraphx::array<int, 1> arr = {5};
    auto* result = migraphx::upper_bound(arr.begin(), arr.end(), 5, migraphx::less{});
    EXPECT(result == arr.end());
}
TEST_CASE(upper_bound_single_element_smaller)
{
    migraphx::array<int, 1> arr = {5};
    auto* result = migraphx::upper_bound(arr.begin(), arr.end(), 3, migraphx::less{});
    EXPECT(result == arr.begin());
}
TEST_CASE(upper_bound_single_element_larger)
{
    migraphx::array<int, 1> arr = {5};
    auto* result = migraphx::upper_bound(arr.begin(), arr.end(), 7, migraphx::less{});
    EXPECT(result == arr.end());
}
TEST_CASE(upper_bound_beginning_duplicates)
{
    migraphx::array<int, 6> arr = {1, 1, 1, 4, 5, 6};
    auto* result = migraphx::upper_bound(arr.begin(), arr.end(), 1, migraphx::less{});
    EXPECT(result == arr.begin() + 3);
}
TEST_CASE(upper_bound_end_duplicates)
{
    migraphx::array<int, 6> arr = {1, 2, 3, 5, 5, 5};
    auto* result = migraphx::upper_bound(arr.begin(), arr.end(), 5, migraphx::less{});
    EXPECT(result == arr.end());
}
TEST_CASE(upper_bound_middle_value)
{
    migraphx::array<int, 7> arr = {1, 3, 5, 7, 9, 11, 13};
    auto* result = migraphx::upper_bound(arr.begin(), arr.end(), 7, migraphx::less{});
    EXPECT(result == arr.begin() + 4);
}
TEST_CASE(upper_bound_partial_range)
{
    migraphx::array<int, 8> arr = {1, 2, 3, 4, 5, 6, 7, 8};
    auto* result = migraphx::upper_bound(arr.begin() + 2, arr.begin() + 6, 4, migraphx::less{});
    EXPECT(result == arr.begin() + 4);
}
TEST_CASE(upper_bound_reverse_comparator)
{
    migraphx::array<int, 5> arr = {5, 4, 3, 2, 1};
    auto* result = migraphx::upper_bound(arr.begin(), arr.end(), 3, migraphx::greater{});
    EXPECT(result == arr.begin() + 3);
}
TEST_CASE(upper_bound_large_array_power_of_two)
{
    migraphx::array<int, 16> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto* result = migraphx::upper_bound(arr.begin(), arr.end(), 8, migraphx::less{});
    EXPECT(result == arr.begin() + 8);
}
TEST_CASE(upper_bound_stress_binary_search)
{
    migraphx::array<int, 15> arr = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
    auto* result = migraphx::upper_bound(arr.begin(), arr.end(), 16, migraphx::less{});
    EXPECT(result == arr.begin() + 8);
}
TEST_CASE(upper_bound_stress_test_boundary)
{
    migraphx::array<int, 7> arr = {1, 2, 3, 4, 5, 6, 7};
    for(int i = 1; i <= 7; ++i)
    {
        auto* result = migraphx::upper_bound(arr.begin(), arr.end(), i, migraphx::less{});
        EXPECT(result == arr.begin() + i);
    }
}

TEST_CASE(sort_basic)
{
    migraphx::array<int, 5> arr = {5, 2, 8, 1, 9};
    migraphx::sort(arr.begin(), arr.end(), migraphx::less{});
    migraphx::array<int, 5> expected = {1, 2, 5, 8, 9};
    EXPECT(arr == expected);
}
TEST_CASE(sort_already_sorted)
{
    migraphx::array<int, 4> arr      = {1, 2, 3, 4};
    migraphx::array<int, 4> original = arr;
    migraphx::sort(arr.begin(), arr.end(), migraphx::less{});
    EXPECT(arr == original);
}
TEST_CASE(sort_reverse_sorted)
{
    migraphx::array<int, 4> arr = {4, 3, 2, 1};
    migraphx::sort(arr.begin(), arr.end(), migraphx::less{});
    migraphx::array<int, 4> expected = {1, 2, 3, 4};
    EXPECT(arr == expected);
}
TEST_CASE(sort_duplicates)
{
    migraphx::array<int, 6> arr = {3, 1, 4, 1, 5, 2};
    migraphx::sort(arr.begin(), arr.end(), migraphx::less{});
    migraphx::array<int, 6> expected = {1, 1, 2, 3, 4, 5};
    EXPECT(arr == expected);
}
TEST_CASE(sort_empty)
{
    empty_range arr = {};
    migraphx::sort(arr.begin(), arr.end(), migraphx::less{});
}
TEST_CASE(sort_single)
{
    migraphx::array<int, 1> arr      = {42};
    migraphx::array<int, 1> original = arr;
    migraphx::sort(arr.begin(), arr.end(), migraphx::less{});
    EXPECT(arr == original);
}
TEST_CASE(sort_default_comparator)
{
    migraphx::array<int, 4> arr = {3, 1, 4, 2};
    migraphx::sort(arr.begin(), arr.end());
    migraphx::array<int, 4> expected = {1, 2, 3, 4};
    EXPECT(arr == expected);
}

TEST_CASE(stable_sort_basic)
{
    migraphx::array<int, 5> arr = {5, 2, 8, 1, 9};
    migraphx::stable_sort(arr.begin(), arr.end(), migraphx::less{});
    migraphx::array<int, 5> expected = {1, 2, 5, 8, 9};
    EXPECT(arr == expected);
}
TEST_CASE(stable_sort_empty)
{
    empty_range arr = {};
    migraphx::stable_sort(arr.begin(), arr.end(), migraphx::less{});
}
TEST_CASE(stable_sort_single)
{
    migraphx::array<int, 1> arr      = {42};
    migraphx::array<int, 1> original = arr;
    migraphx::stable_sort(arr.begin(), arr.end(), migraphx::less{});
    EXPECT(arr == original);
}
TEST_CASE(stable_sort_already_sorted)
{
    migraphx::array<int, 4> arr      = {1, 2, 3, 4};
    migraphx::array<int, 4> original = arr;
    migraphx::stable_sort(arr.begin(), arr.end(), migraphx::less{});
    EXPECT(arr == original);
}
TEST_CASE(stable_sort_default_comparator)
{
    migraphx::array<int, 4> arr = {3, 1, 4, 2};
    migraphx::stable_sort(arr.begin(), arr.end());
    migraphx::array<int, 4> expected = {1, 2, 3, 4};
    EXPECT(arr == expected);
}

TEST_CASE(merge_basic)
{
    migraphx::array<int, 3> arr1   = {1, 3, 5};
    migraphx::array<int, 3> arr2   = {2, 4, 6};
    migraphx::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    migraphx::array<int, 6> expected = {1, 2, 3, 4, 5, 6};
    EXPECT(result == expected);
}
TEST_CASE(merge_empty_first)
{
    empty_range arr1               = {};
    migraphx::array<int, 3> arr2   = {1, 2, 3};
    migraphx::array<int, 3> result = {0, 0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    EXPECT(result == arr2);
}
TEST_CASE(merge_empty_second)
{
    migraphx::array<int, 3> arr1   = {1, 2, 3};
    empty_range arr2               = {};
    migraphx::array<int, 3> result = {0, 0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    EXPECT(result == arr1);
}
TEST_CASE(merge_both_empty)
{
    empty_range arr1   = {};
    empty_range arr2   = {};
    empty_range result = {};
    auto* end_it       = migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    EXPECT(end_it == result.end());
}
TEST_CASE(merge_overlapping_ranges)
{
    migraphx::array<int, 4> arr1   = {1, 3, 5, 7};
    migraphx::array<int, 2> arr2   = {2, 6};
    migraphx::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    migraphx::array<int, 6> expected = {1, 2, 3, 5, 6, 7};
    EXPECT(result == expected);
}
TEST_CASE(merge_duplicates)
{
    migraphx::array<int, 3> arr1   = {1, 2, 3};
    migraphx::array<int, 3> arr2   = {2, 3, 4};
    migraphx::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    migraphx::array<int, 6> expected = {1, 2, 2, 3, 3, 4};
    EXPECT(result == expected);
}
// Additional merge tests (extensive)
TEST_CASE(merge_different_sizes)
{
    migraphx::array<int, 2> arr1   = {1, 5};
    migraphx::array<int, 4> arr2   = {2, 3, 4, 6};
    migraphx::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    migraphx::array<int, 6> expected = {1, 2, 3, 4, 5, 6};
    EXPECT(result == expected);
}
TEST_CASE(merge_first_all_smaller)
{
    migraphx::array<int, 3> arr1   = {1, 2, 3};
    migraphx::array<int, 3> arr2   = {4, 5, 6};
    migraphx::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    migraphx::array<int, 6> expected = {1, 2, 3, 4, 5, 6};
    EXPECT(result == expected);
}
TEST_CASE(merge_first_all_larger)
{
    migraphx::array<int, 3> arr1   = {4, 5, 6};
    migraphx::array<int, 3> arr2   = {1, 2, 3};
    migraphx::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    migraphx::array<int, 6> expected = {1, 2, 3, 4, 5, 6};
    EXPECT(result == expected);
}
TEST_CASE(merge_interleaved)
{
    migraphx::array<int, 4> arr1   = {1, 3, 5, 7};
    migraphx::array<int, 4> arr2   = {2, 4, 6, 8};
    migraphx::array<int, 8> result = {0, 0, 0, 0, 0, 0, 0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    migraphx::array<int, 8> expected = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT(result == expected);
}
TEST_CASE(merge_many_duplicates)
{
    migraphx::array<int, 4> arr1   = {1, 1, 2, 2};
    migraphx::array<int, 4> arr2   = {1, 2, 2, 3};
    migraphx::array<int, 8> result = {0, 0, 0, 0, 0, 0, 0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    migraphx::array<int, 8> expected = {1, 1, 1, 2, 2, 2, 2, 3};
    EXPECT(result == expected);
}
TEST_CASE(merge_all_equal_elements)
{
    migraphx::array<int, 3> arr1   = {5, 5, 5};
    migraphx::array<int, 2> arr2   = {5, 5};
    migraphx::array<int, 5> result = {0, 0, 0, 0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    migraphx::array<int, 5> expected = {5, 5, 5, 5, 5};
    EXPECT(result == expected);
}
TEST_CASE(merge_single_elements)
{
    migraphx::array<int, 1> arr1   = {3};
    migraphx::array<int, 1> arr2   = {1};
    migraphx::array<int, 2> result = {0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    migraphx::array<int, 2> expected = {1, 3};
    EXPECT(result == expected);
}
TEST_CASE(merge_one_single_element)
{
    migraphx::array<int, 1> arr1   = {3};
    migraphx::array<int, 4> arr2   = {1, 2, 4, 5};
    migraphx::array<int, 5> result = {0, 0, 0, 0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    migraphx::array<int, 5> expected = {1, 2, 3, 4, 5};
    EXPECT(result == expected);
}
TEST_CASE(merge_reverse_order)
{
    migraphx::array<int, 3> arr1   = {6, 4, 2};
    migraphx::array<int, 3> arr2   = {5, 3, 1};
    migraphx::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::greater{});
    migraphx::array<int, 6> expected = {6, 5, 4, 3, 2, 1};
    EXPECT(result == expected);
}
TEST_CASE(merge_negative_numbers)
{
    migraphx::array<int, 3> arr1   = {-5, -2, 1};
    migraphx::array<int, 3> arr2   = {-3, 0, 3};
    migraphx::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    migraphx::array<int, 6> expected = {-5, -3, -2, 0, 1, 3};
    EXPECT(result == expected);
}
TEST_CASE(merge_large_difference)
{
    migraphx::array<int, 3> arr1   = {1, 1000, 2000};
    migraphx::array<int, 3> arr2   = {2, 500, 1500};
    migraphx::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    migraphx::array<int, 6> expected = {1, 2, 500, 1000, 1500, 2000};
    EXPECT(result == expected);
}
TEST_CASE(merge_partial_ranges)
{
    migraphx::array<int, 8> arr1   = {0, 1, 2, 3, 4, 5, 6, 7};
    migraphx::array<int, 6> arr2   = {10, 11, 12, 13, 14, 15};
    migraphx::array<int, 5> result = {0, 0, 0, 0, 0};
    migraphx::merge(arr1.begin() + 2,
                    arr1.begin() + 5,
                    arr2.begin() + 1,
                    arr2.begin() + 3,
                    result.begin(),
                    migraphx::less{});
    migraphx::array<int, 5> expected = {2, 3, 4, 11, 12};
    EXPECT(result == expected);
}
TEST_CASE(merge_stability_test)
{
    migraphx::array<int, 4> arr1   = {1, 3, 5, 7};
    migraphx::array<int, 4> arr2   = {2, 3, 6, 7};
    migraphx::array<int, 8> result = {0, 0, 0, 0, 0, 0, 0, 0};
    migraphx::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), migraphx::less{});
    migraphx::array<int, 8> expected = {1, 2, 3, 3, 5, 6, 7, 7};
    EXPECT(result == expected);
}

TEST_CASE(swap_self_assignment)
{
    int x = 42;
    migraphx::swap(x, x);
    EXPECT(x == 42);
}
TEST_CASE(find_if_predicate_consistency)
{
    migraphx::array<int, 4> arr = {1, 2, 3, 4};
    int call_count              = 0;
    auto predicate              = [&call_count](int x) {
        call_count++;
        return x > 2;
    };
    auto* result = migraphx::find_if(arr.begin(), arr.end(), predicate);
    EXPECT(result == arr.begin() + 2);
    EXPECT(call_count == 3);
}
