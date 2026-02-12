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
#include <rocm/algorithm.hpp>
#include <rocm/array.hpp>
#include <migraphx/kernels/float_equal.hpp>
#include <migraphx/kernels/test.hpp>

struct empty_range
{
    constexpr int* begin() { return nullptr; }

    constexpr int* end() { return nullptr; }
};

TEST_CASE(iter_swap_basic)
{
    rocm::array<int, 3> arr = {1, 2, 3};
    rocm::iter_swap(arr.begin(), arr.begin() + 2);
    rocm::array<int, 3> expected = {3, 2, 1};
    EXPECT(arr == expected);
}
TEST_CASE(iter_swap_same_iterator)
{
    rocm::array<int, 3> arr      = {1, 2, 3};
    rocm::array<int, 3> original = arr;
    rocm::iter_swap(arr.begin(), arr.begin());
    EXPECT(arr == original);
}
TEST_CASE(iter_swap_adjacent)
{
    rocm::array<int, 4> arr = {10, 20, 30, 40};
    rocm::iter_swap(arr.begin() + 1, arr.begin() + 2);
    rocm::array<int, 4> expected = {10, 30, 20, 40};
    EXPECT(arr == expected);
}

TEST_CASE(accumulate_basic)
{
    rocm::array<int, 4> arr = {1, 2, 3, 4};
    auto sum = rocm::accumulate(arr.begin(), arr.end(), 0, [](int a, int b) { return a + b; });
    EXPECT(sum == 10);
}
TEST_CASE(accumulate_empty_range)
{
    empty_range arr = {};
    auto sum =
        rocm::accumulate(arr.begin(), arr.begin(), 42, [](int a, int b) { return a + b; });
    EXPECT(sum == 42);
}
TEST_CASE(accumulate_multiply)
{
    rocm::array<int, 3> arr = {2, 3, 4};
    auto product =
        rocm::accumulate(arr.begin(), arr.end(), 1, [](int a, int b) { return a * b; });
    EXPECT(product == 24);
}
TEST_CASE(accumulate_single_element)
{
    rocm::array<int, 1> arr = {7};
    auto sum = rocm::accumulate(arr.begin(), arr.end(), 5, [](int a, int b) { return a + b; });
    EXPECT(sum == 12);
}

TEST_CASE(copy_basic)
{
    rocm::array<int, 4> src = {1, 2, 3, 4};
    rocm::array<int, 4> dst = {0, 0, 0, 0};
    rocm::copy(src.begin(), src.end(), dst.begin());
    EXPECT(src == dst);
}
TEST_CASE(copy_empty_range)
{
    rocm::array<int, 3> src = {1, 2, 3};
    rocm::array<int, 3> dst = {0, 0, 0};
    auto* result                = rocm::copy(src.begin(), src.begin(), dst.begin());
    EXPECT(result == dst.begin());
    rocm::array<int, 3> expected = {0, 0, 0};
    EXPECT(dst == expected);
}
TEST_CASE(copy_partial)
{
    rocm::array<int, 5> src = {10, 20, 30, 40, 50};
    rocm::array<int, 3> dst = {0, 0, 0};
    rocm::copy(src.begin() + 1, src.begin() + 4, dst.begin());
    rocm::array<int, 3> expected = {20, 30, 40};
    EXPECT(dst == expected);
}
TEST_CASE(copy_if_basic)
{
    rocm::array<int, 6> src = {1, 2, 3, 4, 5, 6};
    rocm::array<int, 6> dst = {0, 0, 0, 0, 0, 0};
    auto* end_it =
        rocm::copy_if(src.begin(), src.end(), dst.begin(), [](int x) { return x % 2 == 0; });
    EXPECT(dst[0] == 2);
    EXPECT(dst[1] == 4);
    EXPECT(dst[2] == 6);
    EXPECT(end_it == dst.begin() + 3);
}
TEST_CASE(copy_if_none_match)
{
    rocm::array<int, 3> src = {1, 3, 5};
    rocm::array<int, 3> dst = {0, 0, 0};
    auto* end_it =
        rocm::copy_if(src.begin(), src.end(), dst.begin(), [](int x) { return x % 2 == 0; });
    EXPECT(end_it == dst.begin());
    rocm::array<int, 3> expected = {0, 0, 0};
    EXPECT(dst == expected);
}
TEST_CASE(copy_if_all_match)
{
    rocm::array<int, 3> src = {2, 4, 6};
    rocm::array<int, 3> dst = {0, 0, 0};
    auto* end_it =
        rocm::copy_if(src.begin(), src.end(), dst.begin(), [](int x) { return x % 2 == 0; });
    EXPECT(end_it == dst.end());
    EXPECT(src == dst);
}

TEST_CASE(is_sorted_until_sorted)
{
    rocm::array<int, 4> arr = {1, 2, 3, 4};
    auto* result = rocm::is_sorted_until(arr.begin(), arr.end(), rocm::less<>{});
    EXPECT(result == arr.end());
}
TEST_CASE(is_sorted_until_unsorted)
{
    rocm::array<int, 5> arr = {1, 2, 4, 3, 5};
    auto* result = rocm::is_sorted_until(arr.begin(), arr.end(), rocm::less<>{});
    EXPECT(result == arr.begin() + 3);
}
TEST_CASE(is_sorted_until_empty)
{
    empty_range arr = {};
    auto* result    = rocm::is_sorted_until(arr.begin(), arr.end(), rocm::less<>{});
    EXPECT(result == arr.end());
}
TEST_CASE(is_sorted_until_single)
{
    rocm::array<int, 1> arr = {42};
    auto* result = rocm::is_sorted_until(arr.begin(), arr.end(), rocm::less<>{});
    EXPECT(result == arr.end());
}
TEST_CASE(is_sorted_until_descending)
{
    rocm::array<int, 4> arr = {4, 3, 2, 1};
    auto* result = rocm::is_sorted_until(arr.begin(), arr.end(), rocm::greater<>{});
    EXPECT(result == arr.end());
}
TEST_CASE(is_sorted_true)
{
    rocm::array<int, 4> arr = {1, 2, 3, 4};
    EXPECT(rocm::is_sorted(arr.begin(), arr.end(), rocm::less<>{}));
}
TEST_CASE(is_sorted_false)
{
    rocm::array<int, 4> arr = {1, 3, 2, 4};
    EXPECT(not rocm::is_sorted(arr.begin(), arr.end(), rocm::less<>{}));
}
TEST_CASE(is_sorted_empty)
{
    empty_range arr = {};
    EXPECT(rocm::is_sorted(arr.begin(), arr.end(), rocm::less<>{}));
}
TEST_CASE(is_sorted_duplicates)
{
    rocm::array<int, 4> arr = {1, 2, 2, 3};
    EXPECT(rocm::is_sorted(arr.begin(), arr.end(), rocm::less<>{}));
}

TEST_CASE(for_each_basic)
{
    rocm::array<int, 3> arr = {1, 2, 3};
    int sum                     = 0;
    rocm::for_each(arr.begin(), arr.end(), [&sum](int x) { sum += x; });
    EXPECT(sum == 6);
}
TEST_CASE(for_each_modify)
{
    rocm::array<int, 3> arr = {1, 2, 3};
    rocm::for_each(arr.begin(), arr.end(), [](int& x) { x *= 2; });
    rocm::array<int, 3> expected = {2, 4, 6};
    EXPECT(arr == expected);
}
TEST_CASE(for_each_empty)
{
    empty_range arr = {};
    int count       = 0;
    rocm::for_each(arr.begin(), arr.end(), [&count](int) { count++; });
    EXPECT(count == 0);
}

TEST_CASE(find_if_found)
{
    rocm::array<int, 5> arr = {1, 3, 5, 7, 9};
    auto* result = rocm::find_if(arr.begin(), arr.end(), [](int x) { return x > 5; });
    EXPECT(result == arr.begin() + 3);
}
TEST_CASE(find_if_not_found)
{
    rocm::array<int, 4> arr = {1, 3, 5, 7};
    auto* result = rocm::find_if(arr.begin(), arr.end(), [](int x) { return x > 10; });
    EXPECT(result == arr.end());
}
TEST_CASE(find_if_first_element)
{
    rocm::array<int, 4> arr = {10, 3, 5, 7};
    auto* result = rocm::find_if(arr.begin(), arr.end(), [](int x) { return x > 5; });
    EXPECT(result == arr.begin());
}
TEST_CASE(find_if_empty)
{
    empty_range arr = {};
    auto* result    = rocm::find_if(arr.begin(), arr.end(), [](int x) { return x > 0; });
    EXPECT(result == arr.end());
}
TEST_CASE(find_basic)
{
    rocm::array<int, 4> arr = {10, 20, 30, 40};
    auto* result                = rocm::find(arr.begin(), arr.end(), 30);
    EXPECT(result == arr.begin() + 2);
}
TEST_CASE(find_not_found)
{
    rocm::array<int, 4> arr = {10, 20, 30, 40};
    auto* result                = rocm::find(arr.begin(), arr.end(), 50);
    EXPECT(result == arr.end());
}
TEST_CASE(find_first_element)
{
    rocm::array<int, 4> arr = {10, 20, 30, 40};
    auto* result                = rocm::find(arr.begin(), arr.end(), 10);
    EXPECT(result == arr.begin());
}
TEST_CASE(find_duplicates)
{
    rocm::array<int, 5> arr = {10, 20, 30, 20, 40};
    auto* result                = rocm::find(arr.begin(), arr.end(), 20);
    EXPECT(result == arr.begin() + 1);
}

TEST_CASE(any_of_true)
{
    rocm::array<int, 4> arr = {1, 3, 5, 7};
    EXPECT(rocm::any_of(arr.begin(), arr.end(), [](int x) { return x > 5; }));
}
TEST_CASE(any_of_false)
{
    rocm::array<int, 4> arr = {1, 3, 5, 7};
    EXPECT(not rocm::any_of(arr.begin(), arr.end(), [](int x) { return x > 10; }));
}
TEST_CASE(any_of_empty)
{
    empty_range arr = {};
    EXPECT(not rocm::any_of(arr.begin(), arr.end(), [](int x) { return x > 0; }));
}
TEST_CASE(any_of_first_element)
{
    rocm::array<int, 4> arr = {10, 1, 2, 3};
    EXPECT(rocm::any_of(arr.begin(), arr.end(), [](int x) { return x > 5; }));
}
TEST_CASE(none_of_true)
{
    rocm::array<int, 4> arr = {1, 3, 5, 7};
    EXPECT(rocm::none_of(arr.begin(), arr.end(), [](int x) { return x > 10; }));
}
TEST_CASE(none_of_false)
{
    rocm::array<int, 4> arr = {1, 3, 5, 7};
    EXPECT(not rocm::none_of(arr.begin(), arr.end(), [](int x) { return x > 5; }));
}
TEST_CASE(none_of_empty)
{
    empty_range arr = {};
    EXPECT(rocm::none_of(arr.begin(), arr.end(), [](int x) { return x > 0; }));
}
TEST_CASE(all_of_true)
{
    rocm::array<int, 4> arr = {2, 4, 6, 8};
    EXPECT(rocm::all_of(arr.begin(), arr.end(), [](int x) { return x % 2 == 0; }));
}
TEST_CASE(all_of_false)
{
    rocm::array<int, 4> arr = {2, 4, 5, 8};
    EXPECT(not rocm::all_of(arr.begin(), arr.end(), [](int x) { return x % 2 == 0; }));
}
TEST_CASE(all_of_empty)
{
    empty_range arr = {};
    EXPECT(rocm::all_of(arr.begin(), arr.end(), [](int x) { return x > 0; }));
}
TEST_CASE(all_of_single_true)
{
    rocm::array<int, 1> arr = {4};
    EXPECT(rocm::all_of(arr.begin(), arr.end(), [](int x) { return x % 2 == 0; }));
}
TEST_CASE(all_of_single_false)
{
    rocm::array<int, 1> arr = {5};
    EXPECT(not rocm::all_of(arr.begin(), arr.end(), [](int x) { return x % 2 == 0; }));
}

TEST_CASE(search_found)
{
    rocm::array<int, 6> haystack = {1, 2, 3, 4, 3, 4};
    rocm::array<int, 2> needle   = {3, 4};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 2);
}
TEST_CASE(search_not_found)
{
    rocm::array<int, 5> haystack = {1, 2, 3, 4, 5};
    rocm::array<int, 2> needle   = {6, 7};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.end());
}
TEST_CASE(search_empty_needle)
{
    rocm::array<int, 3> haystack = {1, 2, 3};
    empty_range needle               = {};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin());
}
TEST_CASE(search_empty_haystack)
{
    empty_range haystack           = {};
    rocm::array<int, 1> needle = {1};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.end());
}
TEST_CASE(search_exact_match)
{
    rocm::array<int, 3> haystack = {1, 2, 3};
    rocm::array<int, 3> needle   = {1, 2, 3};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin());
}
TEST_CASE(search_partial_match)
{
    rocm::array<int, 5> haystack = {1, 2, 1, 2, 3};
    rocm::array<int, 3> needle   = {1, 2, 3};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 2);
}
// Additional search tests (extensive)
TEST_CASE(search_overlapping_pattern)
{
    rocm::array<int, 8> haystack = {1, 2, 1, 2, 1, 2, 3, 4};
    rocm::array<int, 3> needle   = {1, 2, 3};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 4);
}
TEST_CASE(search_pattern_at_end)
{
    rocm::array<int, 6> haystack = {1, 2, 3, 4, 5, 6};
    rocm::array<int, 2> needle   = {5, 6};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 4);
}
TEST_CASE(search_pattern_at_beginning)
{
    rocm::array<int, 6> haystack = {1, 2, 3, 4, 5, 6};
    rocm::array<int, 2> needle   = {1, 2};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin());
}
TEST_CASE(search_single_element_pattern)
{
    rocm::array<int, 5> haystack = {1, 3, 5, 7, 9};
    rocm::array<int, 1> needle   = {5};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 2);
}
TEST_CASE(search_single_element_pattern_not_found)
{
    rocm::array<int, 5> haystack = {1, 3, 5, 7, 9};
    rocm::array<int, 1> needle   = {4};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.end());
}
TEST_CASE(search_repeated_elements)
{
    rocm::array<int, 8> haystack = {2, 2, 2, 2, 1, 2, 2, 3};
    rocm::array<int, 3> needle   = {2, 2, 3};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 5);
}
TEST_CASE(search_partial_match_backtrack)
{
    rocm::array<int, 10> haystack = {1, 2, 3, 1, 2, 4, 1, 2, 3, 4};
    rocm::array<int, 4> needle    = {1, 2, 3, 4};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 6);
}
TEST_CASE(search_multiple_false_starts)
{
    rocm::array<int, 12> haystack = {1, 2, 1, 2, 1, 2, 1, 2, 3, 4, 5, 6};
    rocm::array<int, 4> needle    = {1, 2, 3, 4};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 6);
}
TEST_CASE(search_needle_longer_than_haystack)
{
    rocm::array<int, 3> haystack = {1, 2, 3};
    rocm::array<int, 5> needle   = {1, 2, 3, 4, 5};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.end());
}
TEST_CASE(search_identical_arrays)
{
    rocm::array<int, 4> haystack = {5, 10, 15, 20};
    rocm::array<int, 4> needle   = {5, 10, 15, 20};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin());
}
TEST_CASE(search_all_same_elements)
{
    rocm::array<int, 6> haystack = {3, 3, 3, 3, 3, 3};
    rocm::array<int, 3> needle   = {3, 3, 3};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin());
}
TEST_CASE(search_pattern_appears_multiple_times)
{
    rocm::array<int, 9> haystack = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    rocm::array<int, 2> needle   = {2, 3};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 1);
}
TEST_CASE(search_partial_range)
{
    rocm::array<int, 10> haystack = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    rocm::array<int, 2> needle    = {4, 5};
    auto* result =
        rocm::search(haystack.begin() + 2, haystack.begin() + 7, needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 4);
}
TEST_CASE(search_stress_long_pattern)
{
    rocm::array<int, 15> haystack = {1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1};
    rocm::array<int, 6> needle    = {2, 3, 4, 5, 6, 7};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 5);
}
TEST_CASE(search_stress_worst_case_complexity)
{
    rocm::array<int, 8> haystack = {1, 1, 1, 1, 1, 1, 1, 2};
    rocm::array<int, 7> needle   = {1, 1, 1, 1, 1, 1, 2};
    auto* result = rocm::search(haystack.begin(), haystack.end(), needle.begin(), needle.end());
    EXPECT(result == haystack.begin() + 1);
}

TEST_CASE(inner_product_basic)
{
    rocm::array<int, 3> a = {1, 2, 3};
    rocm::array<int, 3> b = {4, 5, 6};
    auto result               = rocm::inner_product(a.begin(), a.end(), b.begin(), 0);
    EXPECT(result == 32);
}
TEST_CASE(inner_product_custom_ops)
{
    rocm::array<int, 3> a = {1, 2, 3};
    rocm::array<int, 3> b = {4, 5, 6};
    auto result               = rocm::inner_product(
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
    auto result   = rocm::inner_product(a.begin(), a.end(), b.begin(), 42);
    EXPECT(result == 42);
}
TEST_CASE(inner_product_single)
{
    rocm::array<int, 1> a = {3};
    rocm::array<int, 1> b = {7};
    auto result               = rocm::inner_product(a.begin(), a.end(), b.begin(), 5);
    EXPECT(result == 26);
}

TEST_CASE(equal_true)
{
    rocm::array<int, 4> a = {1, 2, 3, 4};
    rocm::array<int, 4> b = {1, 2, 3, 4};
    EXPECT(rocm::equal(a.begin(), a.end(), b.begin(), [](int x, int y) { return x == y; }));
}
TEST_CASE(equal_false)
{
    rocm::array<int, 4> a = {1, 2, 3, 4};
    rocm::array<int, 4> b = {1, 2, 3, 5};
    EXPECT(not rocm::equal(a.begin(), a.end(), b.begin(), [](int x, int y) { return x == y; }));
}
TEST_CASE(equal_empty)
{
    empty_range a = {};
    empty_range b = {};
    EXPECT(rocm::equal(a.begin(), a.end(), b.begin(), [](int x, int y) { return x == y; }));
}
TEST_CASE(equal_custom_predicate)
{
    rocm::array<int, 3> a = {1, 2, 3};
    rocm::array<int, 3> b = {2, 4, 6};
    EXPECT(rocm::equal(a.begin(), a.end(), b.begin(), [](int x, int y) { return x * 2 == y; }));
}

TEST_CASE(iota_basic)
{
    rocm::array<int, 5> arr = {0, 0, 0, 0, 0};
    rocm::iota(arr.begin(), arr.end(), 10);
    rocm::array<int, 5> expected = {10, 11, 12, 13, 14};
    EXPECT(arr == expected);
}
TEST_CASE(iota_empty)
{
    empty_range arr = {};
    rocm::iota(arr.begin(), arr.end(), 5);
}
TEST_CASE(iota_single)
{
    rocm::array<int, 1> arr = {0};
    rocm::iota(arr.begin(), arr.end(), 42);
    EXPECT(arr[0] == 42);
}
TEST_CASE(iota_negative)
{
    rocm::array<int, 3> arr = {0, 0, 0};
    rocm::iota(arr.begin(), arr.end(), -5);
    rocm::array<int, 3> expected = {-5, -4, -3};
    EXPECT(arr == expected);
}

TEST_CASE(min_element_basic)
{
    rocm::array<int, 5> arr = {3, 1, 4, 1, 5};
    auto* result                = rocm::min_element(arr.begin(), arr.end(), rocm::less<>{});
    EXPECT(result == arr.begin() + 1);
}
TEST_CASE(min_element_empty)
{
    empty_range arr = {};
    auto* result    = rocm::min_element(arr.begin(), arr.end(), rocm::less<>{});
    EXPECT(result == arr.end());
}
TEST_CASE(min_element_single)
{
    rocm::array<int, 1> arr = {42};
    auto* result                = rocm::min_element(arr.begin(), arr.end(), rocm::less<>{});
    EXPECT(result == arr.begin());
}
TEST_CASE(min_element_all_equal)
{
    rocm::array<int, 4> arr = {5, 5, 5, 5};
    auto* result                = rocm::min_element(arr.begin(), arr.end(), rocm::less<>{});
    EXPECT(result == arr.begin());
}
TEST_CASE(min_element_custom_compare)
{
    rocm::array<int, 4> arr = {1, 2, 3, 4};
    auto* result = rocm::min_element(arr.begin(), arr.end(), rocm::greater<>{});
    EXPECT(result == arr.begin() + 3);
}

TEST_CASE(rotate_left_by_two)
{
    rocm::array<int, 6> arr      = {1, 2, 3, 4, 5, 6};
    auto* result                     = rocm::rotate(arr.begin(), arr.begin() + 2, arr.end());
    rocm::array<int, 6> expected = {3, 4, 5, 6, 1, 2};
    EXPECT(arr == expected);
    EXPECT(result == arr.begin() + 4);
}
TEST_CASE(rotate_right_by_one)
{
    rocm::array<int, 5> arr      = {1, 2, 3, 4, 5};
    auto* result                     = rocm::rotate(arr.begin(), arr.end() - 1, arr.end());
    rocm::array<int, 5> expected = {5, 1, 2, 3, 4};
    EXPECT(arr == expected);
    EXPECT(result == arr.begin() + 1);
}
TEST_CASE(rotate_half_array)
{
    rocm::array<int, 8> arr      = {1, 2, 3, 4, 5, 6, 7, 8};
    auto* result                     = rocm::rotate(arr.begin(), arr.begin() + 4, arr.end());
    rocm::array<int, 8> expected = {5, 6, 7, 8, 1, 2, 3, 4};
    EXPECT(arr == expected);
    EXPECT(result == arr.begin() + 4);
}
TEST_CASE(rotate_almost_full)
{
    rocm::array<int, 5> arr      = {1, 2, 3, 4, 5};
    auto* result                     = rocm::rotate(arr.begin(), arr.begin() + 4, arr.end());
    rocm::array<int, 5> expected = {5, 1, 2, 3, 4};
    EXPECT(arr == expected);
    EXPECT(result == arr.begin() + 1);
}
TEST_CASE(rotate_two_elements)
{
    rocm::array<int, 2> arr      = {1, 2};
    auto* result                     = rocm::rotate(arr.begin(), arr.begin() + 1, arr.end());
    rocm::array<int, 2> expected = {2, 1};
    EXPECT(arr == expected);
    EXPECT(result == arr.begin() + 1);
}
TEST_CASE(rotate_partial_range)
{
    rocm::array<int, 7> arr = {1, 2, 3, 4, 5, 6, 7};
    auto* result = rocm::rotate(arr.begin() + 1, arr.begin() + 3, arr.begin() + 5);
    rocm::array<int, 7> expected = {1, 4, 5, 2, 3, 6, 7};
    EXPECT(arr == expected);
    EXPECT(result == arr.begin() + 3);
}
TEST_CASE(rotate_with_duplicates)
{
    rocm::array<int, 6> arr      = {1, 1, 2, 2, 3, 3};
    auto* result                     = rocm::rotate(arr.begin(), arr.begin() + 2, arr.end());
    rocm::array<int, 6> expected = {2, 2, 3, 3, 1, 1};
    EXPECT(arr == expected);
    EXPECT(result == arr.begin() + 4);
}
TEST_CASE(rotate_stress_test_large_shift)
{
    rocm::array<int, 10> arr      = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    auto* result                      = rocm::rotate(arr.begin(), arr.begin() + 7, arr.end());
    rocm::array<int, 10> expected = {7, 8, 9, 0, 1, 2, 3, 4, 5, 6};
    EXPECT(arr == expected);
    EXPECT(result == arr.begin() + 3);
}
TEST_CASE(rotate_edge_case_middle_equals_first)
{
    rocm::array<int, 4> arr      = {1, 2, 3, 4};
    rocm::array<int, 4> original = arr;
    auto* result                     = rocm::rotate(arr.begin(), arr.begin(), arr.end());
    EXPECT(arr == original);
    EXPECT(result == arr.end());
}
TEST_CASE(rotate_edge_case_middle_equals_last)
{
    rocm::array<int, 4> arr      = {1, 2, 3, 4};
    rocm::array<int, 4> original = arr;
    auto* result                     = rocm::rotate(arr.begin(), arr.end(), arr.end());
    EXPECT(arr == original);
    EXPECT(result == arr.begin());
}

TEST_CASE(upper_bound_basic)
{
    rocm::array<int, 5> arr = {1, 2, 2, 3, 4};
    auto* result = rocm::upper_bound(arr.begin(), arr.end(), 2, rocm::less<>{});
    EXPECT(result == arr.begin() + 3);
}
TEST_CASE(upper_bound_not_found)
{
    rocm::array<int, 4> arr = {1, 2, 3, 4};
    auto* result = rocm::upper_bound(arr.begin(), arr.end(), 5, rocm::less<>{});
    EXPECT(result == arr.end());
}
TEST_CASE(upper_bound_first_element)
{
    rocm::array<int, 4> arr = {1, 2, 3, 4};
    auto* result = rocm::upper_bound(arr.begin(), arr.end(), 0, rocm::less<>{});
    EXPECT(result == arr.begin());
}
TEST_CASE(upper_bound_empty)
{
    empty_range arr = {};
    auto* result    = rocm::upper_bound(arr.begin(), arr.end(), 5, rocm::less<>{});
    EXPECT(result == arr.end());
}
TEST_CASE(upper_bound_all_equal)
{
    rocm::array<int, 3> arr = {2, 2, 2};
    auto* result = rocm::upper_bound(arr.begin(), arr.end(), 2, rocm::less<>{});
    EXPECT(result == arr.end());
}
TEST_CASE(upper_bound_multiple_duplicates)
{
    rocm::array<int, 8> arr = {1, 2, 2, 2, 2, 3, 4, 5};
    auto* result = rocm::upper_bound(arr.begin(), arr.end(), 2, rocm::less<>{});
    EXPECT(result == arr.begin() + 5);
}
TEST_CASE(upper_bound_all_smaller)
{
    rocm::array<int, 5> arr = {1, 2, 3, 4, 5};
    auto* result = rocm::upper_bound(arr.begin(), arr.end(), 0, rocm::less<>{});
    EXPECT(result == arr.begin());
}
TEST_CASE(upper_bound_all_larger)
{
    rocm::array<int, 5> arr = {1, 2, 3, 4, 5};
    auto* result = rocm::upper_bound(arr.begin(), arr.end(), 10, rocm::less<>{});
    EXPECT(result == arr.end());
}
TEST_CASE(upper_bound_single_element_match)
{
    rocm::array<int, 1> arr = {5};
    auto* result = rocm::upper_bound(arr.begin(), arr.end(), 5, rocm::less<>{});
    EXPECT(result == arr.end());
}
TEST_CASE(upper_bound_single_element_smaller)
{
    rocm::array<int, 1> arr = {5};
    auto* result = rocm::upper_bound(arr.begin(), arr.end(), 3, rocm::less<>{});
    EXPECT(result == arr.begin());
}
TEST_CASE(upper_bound_single_element_larger)
{
    rocm::array<int, 1> arr = {5};
    auto* result = rocm::upper_bound(arr.begin(), arr.end(), 7, rocm::less<>{});
    EXPECT(result == arr.end());
}
TEST_CASE(upper_bound_beginning_duplicates)
{
    rocm::array<int, 6> arr = {1, 1, 1, 4, 5, 6};
    auto* result = rocm::upper_bound(arr.begin(), arr.end(), 1, rocm::less<>{});
    EXPECT(result == arr.begin() + 3);
}
TEST_CASE(upper_bound_end_duplicates)
{
    rocm::array<int, 6> arr = {1, 2, 3, 5, 5, 5};
    auto* result = rocm::upper_bound(arr.begin(), arr.end(), 5, rocm::less<>{});
    EXPECT(result == arr.end());
}
TEST_CASE(upper_bound_middle_value)
{
    rocm::array<int, 7> arr = {1, 3, 5, 7, 9, 11, 13};
    auto* result = rocm::upper_bound(arr.begin(), arr.end(), 7, rocm::less<>{});
    EXPECT(result == arr.begin() + 4);
}
TEST_CASE(upper_bound_partial_range)
{
    rocm::array<int, 8> arr = {1, 2, 3, 4, 5, 6, 7, 8};
    auto* result = rocm::upper_bound(arr.begin() + 2, arr.begin() + 6, 4, rocm::less<>{});
    EXPECT(result == arr.begin() + 4);
}
TEST_CASE(upper_bound_reverse_comparator)
{
    rocm::array<int, 5> arr = {5, 4, 3, 2, 1};
    auto* result = rocm::upper_bound(arr.begin(), arr.end(), 3, rocm::greater<>{});
    EXPECT(result == arr.begin() + 3);
}
TEST_CASE(upper_bound_large_array_power_of_two)
{
    rocm::array<int, 16> arr = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    auto* result = rocm::upper_bound(arr.begin(), arr.end(), 8, rocm::less<>{});
    EXPECT(result == arr.begin() + 8);
}
TEST_CASE(upper_bound_stress_binary_search)
{
    rocm::array<int, 15> arr = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30};
    auto* result = rocm::upper_bound(arr.begin(), arr.end(), 16, rocm::less<>{});
    EXPECT(result == arr.begin() + 8);
}
TEST_CASE(upper_bound_stress_test_boundary)
{
    rocm::array<int, 7> arr = {1, 2, 3, 4, 5, 6, 7};
    for(int i = 1; i <= 7; ++i)
    {
        auto* result = rocm::upper_bound(arr.begin(), arr.end(), i, rocm::less<>{});
        EXPECT(result == arr.begin() + i);
    }
}

TEST_CASE(sort_basic)
{
    rocm::array<int, 5> arr = {5, 2, 8, 1, 9};
    rocm::sort(arr.begin(), arr.end(), rocm::less<>{});
    rocm::array<int, 5> expected = {1, 2, 5, 8, 9};
    EXPECT(arr == expected);
}
TEST_CASE(sort_already_sorted)
{
    rocm::array<int, 4> arr      = {1, 2, 3, 4};
    rocm::array<int, 4> original = arr;
    rocm::sort(arr.begin(), arr.end(), rocm::less<>{});
    EXPECT(arr == original);
}
TEST_CASE(sort_reverse_sorted)
{
    rocm::array<int, 4> arr = {4, 3, 2, 1};
    rocm::sort(arr.begin(), arr.end(), rocm::less<>{});
    rocm::array<int, 4> expected = {1, 2, 3, 4};
    EXPECT(arr == expected);
}
TEST_CASE(sort_duplicates)
{
    rocm::array<int, 6> arr = {3, 1, 4, 1, 5, 2};
    rocm::sort(arr.begin(), arr.end(), rocm::less<>{});
    rocm::array<int, 6> expected = {1, 1, 2, 3, 4, 5};
    EXPECT(arr == expected);
}
TEST_CASE(sort_empty)
{
    empty_range arr = {};
    rocm::sort(arr.begin(), arr.end(), rocm::less<>{});
}
TEST_CASE(sort_single)
{
    rocm::array<int, 1> arr      = {42};
    rocm::array<int, 1> original = arr;
    rocm::sort(arr.begin(), arr.end(), rocm::less<>{});
    EXPECT(arr == original);
}
TEST_CASE(sort_default_comparator)
{
    rocm::array<int, 4> arr = {3, 1, 4, 2};
    rocm::sort(arr.begin(), arr.end());
    rocm::array<int, 4> expected = {1, 2, 3, 4};
    EXPECT(arr == expected);
}

TEST_CASE(stable_sort_basic)
{
    rocm::array<int, 5> arr = {5, 2, 8, 1, 9};
    rocm::stable_sort(arr.begin(), arr.end(), rocm::less<>{});
    rocm::array<int, 5> expected = {1, 2, 5, 8, 9};
    EXPECT(arr == expected);
}
TEST_CASE(stable_sort_empty)
{
    empty_range arr = {};
    rocm::stable_sort(arr.begin(), arr.end(), rocm::less<>{});
}
TEST_CASE(stable_sort_single)
{
    rocm::array<int, 1> arr      = {42};
    rocm::array<int, 1> original = arr;
    rocm::stable_sort(arr.begin(), arr.end(), rocm::less<>{});
    EXPECT(arr == original);
}
TEST_CASE(stable_sort_already_sorted)
{
    rocm::array<int, 4> arr      = {1, 2, 3, 4};
    rocm::array<int, 4> original = arr;
    rocm::stable_sort(arr.begin(), arr.end(), rocm::less<>{});
    EXPECT(arr == original);
}
TEST_CASE(stable_sort_default_comparator)
{
    rocm::array<int, 4> arr = {3, 1, 4, 2};
    rocm::stable_sort(arr.begin(), arr.end());
    rocm::array<int, 4> expected = {1, 2, 3, 4};
    EXPECT(arr == expected);
}

TEST_CASE(merge_basic)
{
    rocm::array<int, 3> arr1   = {1, 3, 5};
    rocm::array<int, 3> arr2   = {2, 4, 6};
    rocm::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    rocm::array<int, 6> expected = {1, 2, 3, 4, 5, 6};
    EXPECT(result == expected);
}
TEST_CASE(merge_empty_first)
{
    empty_range arr1               = {};
    rocm::array<int, 3> arr2   = {1, 2, 3};
    rocm::array<int, 3> result = {0, 0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    EXPECT(result == arr2);
}
TEST_CASE(merge_empty_second)
{
    rocm::array<int, 3> arr1   = {1, 2, 3};
    empty_range arr2               = {};
    rocm::array<int, 3> result = {0, 0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    EXPECT(result == arr1);
}
TEST_CASE(merge_both_empty)
{
    empty_range arr1   = {};
    empty_range arr2   = {};
    empty_range result = {};
    auto* end_it       = rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    EXPECT(end_it == result.end());
}
TEST_CASE(merge_overlapping_ranges)
{
    rocm::array<int, 4> arr1   = {1, 3, 5, 7};
    rocm::array<int, 2> arr2   = {2, 6};
    rocm::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    rocm::array<int, 6> expected = {1, 2, 3, 5, 6, 7};
    EXPECT(result == expected);
}
TEST_CASE(merge_duplicates)
{
    rocm::array<int, 3> arr1   = {1, 2, 3};
    rocm::array<int, 3> arr2   = {2, 3, 4};
    rocm::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    rocm::array<int, 6> expected = {1, 2, 2, 3, 3, 4};
    EXPECT(result == expected);
}
// Additional merge tests (extensive)
TEST_CASE(merge_different_sizes)
{
    rocm::array<int, 2> arr1   = {1, 5};
    rocm::array<int, 4> arr2   = {2, 3, 4, 6};
    rocm::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    rocm::array<int, 6> expected = {1, 2, 3, 4, 5, 6};
    EXPECT(result == expected);
}
TEST_CASE(merge_first_all_smaller)
{
    rocm::array<int, 3> arr1   = {1, 2, 3};
    rocm::array<int, 3> arr2   = {4, 5, 6};
    rocm::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    rocm::array<int, 6> expected = {1, 2, 3, 4, 5, 6};
    EXPECT(result == expected);
}
TEST_CASE(merge_first_all_larger)
{
    rocm::array<int, 3> arr1   = {4, 5, 6};
    rocm::array<int, 3> arr2   = {1, 2, 3};
    rocm::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    rocm::array<int, 6> expected = {1, 2, 3, 4, 5, 6};
    EXPECT(result == expected);
}
TEST_CASE(merge_interleaved)
{
    rocm::array<int, 4> arr1   = {1, 3, 5, 7};
    rocm::array<int, 4> arr2   = {2, 4, 6, 8};
    rocm::array<int, 8> result = {0, 0, 0, 0, 0, 0, 0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    rocm::array<int, 8> expected = {1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT(result == expected);
}
TEST_CASE(merge_many_duplicates)
{
    rocm::array<int, 4> arr1   = {1, 1, 2, 2};
    rocm::array<int, 4> arr2   = {1, 2, 2, 3};
    rocm::array<int, 8> result = {0, 0, 0, 0, 0, 0, 0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    rocm::array<int, 8> expected = {1, 1, 1, 2, 2, 2, 2, 3};
    EXPECT(result == expected);
}
TEST_CASE(merge_all_equal_elements)
{
    rocm::array<int, 3> arr1   = {5, 5, 5};
    rocm::array<int, 2> arr2   = {5, 5};
    rocm::array<int, 5> result = {0, 0, 0, 0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    rocm::array<int, 5> expected = {5, 5, 5, 5, 5};
    EXPECT(result == expected);
}
TEST_CASE(merge_single_elements)
{
    rocm::array<int, 1> arr1   = {3};
    rocm::array<int, 1> arr2   = {1};
    rocm::array<int, 2> result = {0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    rocm::array<int, 2> expected = {1, 3};
    EXPECT(result == expected);
}
TEST_CASE(merge_one_single_element)
{
    rocm::array<int, 1> arr1   = {3};
    rocm::array<int, 4> arr2   = {1, 2, 4, 5};
    rocm::array<int, 5> result = {0, 0, 0, 0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    rocm::array<int, 5> expected = {1, 2, 3, 4, 5};
    EXPECT(result == expected);
}
TEST_CASE(merge_reverse_order)
{
    rocm::array<int, 3> arr1   = {6, 4, 2};
    rocm::array<int, 3> arr2   = {5, 3, 1};
    rocm::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::greater<>{});
    rocm::array<int, 6> expected = {6, 5, 4, 3, 2, 1};
    EXPECT(result == expected);
}
TEST_CASE(merge_negative_numbers)
{
    rocm::array<int, 3> arr1   = {-5, -2, 1};
    rocm::array<int, 3> arr2   = {-3, 0, 3};
    rocm::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    rocm::array<int, 6> expected = {-5, -3, -2, 0, 1, 3};
    EXPECT(result == expected);
}
TEST_CASE(merge_large_difference)
{
    rocm::array<int, 3> arr1   = {1, 1000, 2000};
    rocm::array<int, 3> arr2   = {2, 500, 1500};
    rocm::array<int, 6> result = {0, 0, 0, 0, 0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    rocm::array<int, 6> expected = {1, 2, 500, 1000, 1500, 2000};
    EXPECT(result == expected);
}
TEST_CASE(merge_partial_ranges)
{
    rocm::array<int, 8> arr1   = {0, 1, 2, 3, 4, 5, 6, 7};
    rocm::array<int, 6> arr2   = {10, 11, 12, 13, 14, 15};
    rocm::array<int, 5> result = {0, 0, 0, 0, 0};
    rocm::merge(arr1.begin() + 2,
                    arr1.begin() + 5,
                    arr2.begin() + 1,
                    arr2.begin() + 3,
                    result.begin(),
                    rocm::less<>{});
    rocm::array<int, 5> expected = {2, 3, 4, 11, 12};
    EXPECT(result == expected);
}
TEST_CASE(merge_stability_test)
{
    rocm::array<int, 4> arr1   = {1, 3, 5, 7};
    rocm::array<int, 4> arr2   = {2, 3, 6, 7};
    rocm::array<int, 8> result = {0, 0, 0, 0, 0, 0, 0, 0};
    rocm::merge(
        arr1.begin(), arr1.end(), arr2.begin(), arr2.end(), result.begin(), rocm::less<>{});
    rocm::array<int, 8> expected = {1, 2, 3, 3, 5, 6, 7, 7};
    EXPECT(result == expected);
}

TEST_CASE(swap_self_assignment)
{
    int x = 42;
    rocm::swap(x, x);
    EXPECT(x == 42);
}
TEST_CASE(find_if_predicate_consistency)
{
    rocm::array<int, 4> arr = {1, 2, 3, 4};
    int call_count              = 0;
    auto predicate              = [&call_count](int x) {
        call_count++;
        return x > 2;
    };
    auto* result = rocm::find_if(arr.begin(), arr.end(), predicate);
    EXPECT(result == arr.begin() + 2);
    EXPECT(call_count == 3);
}
