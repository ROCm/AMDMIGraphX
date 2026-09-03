/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocm/iterator/reverse_iterator.hpp>
#include <migraphx/kernels/test.hpp>

struct point
{
    int x;
    int y;
};

// A class-type iterator, which is not implicitly convertible to a pointer
template <class T>
struct wrapper_iterator
{
    using difference_type   = rocm::ptrdiff_t;
    using value_type        = rocm::remove_cv_t<T>;
    using pointer           = T*;
    using reference         = T&;
    using iterator_category = rocm::random_access_iterator_tag;

    T* p = nullptr;

    constexpr reference operator*() const { return *p; }
    constexpr pointer operator->() const { return p; }

    constexpr wrapper_iterator& operator++()
    {
        ++p;
        return *this;
    }
    constexpr wrapper_iterator& operator--()
    {
        --p;
        return *this;
    }
    constexpr wrapper_iterator& operator+=(difference_type n)
    {
        p += n;
        return *this;
    }
    constexpr wrapper_iterator& operator-=(difference_type n)
    {
        p -= n;
        return *this;
    }
    friend constexpr wrapper_iterator operator+(wrapper_iterator it, difference_type n)
    {
        return it += n;
    }
    friend constexpr wrapper_iterator operator-(wrapper_iterator it, difference_type n)
    {
        return it -= n;
    }
    friend constexpr difference_type operator-(wrapper_iterator x, wrapper_iterator y)
    {
        return x.p - y.p;
    }
    friend constexpr bool operator==(wrapper_iterator x, wrapper_iterator y) { return x.p == y.p; }
    friend constexpr bool operator!=(wrapper_iterator x, wrapper_iterator y) { return x.p != y.p; }
};

// ---- Construction ----

TEST_CASE(default_construct)
{
    constexpr rocm::reverse_iterator<const int*> ri{};
    EXPECT(ri.base() == nullptr);
}

TEST_CASE(construct_from_pointer)
{
    int arr[] = {10, 20, 30};
    rocm::reverse_iterator<const int*> ri(arr + 3);
    EXPECT(ri.base() == arr + 3);
}

TEST_CASE(copy_construct)
{
    int arr[] = {1, 2, 3};
    rocm::reverse_iterator<const int*> ri1(arr + 3);
    rocm::reverse_iterator<const int*> ri2(ri1);
    EXPECT(ri2.base() == arr + 3);
}

// ---- Dereference ----

TEST_CASE(dereference)
{
    int arr[] = {10, 20, 30};
    rocm::reverse_iterator<const int*> ri(arr + 3);
    EXPECT(*ri == 30);
}

TEST_CASE(dereference_middle)
{
    int arr[] = {10, 20, 30};
    rocm::reverse_iterator<const int*> ri(arr + 2);
    EXPECT(*ri == 20);
}

TEST_CASE(dereference_first)
{
    int arr[] = {10, 20, 30};
    rocm::reverse_iterator<const int*> ri(arr + 1);
    EXPECT(*ri == 10);
}

// ---- Arrow ----

TEST_CASE(arrow_pointer)
{
    point arr[] = {{1, 2}, {3, 4}, {5, 6}};
    rocm::reverse_iterator<const point*> ri(arr + 3);
    EXPECT(ri->x == 5);
    EXPECT(ri->y == 6);
}

TEST_CASE(arrow_pointer_traversal)
{
    point arr[] = {{1, 2}, {3, 4}, {5, 6}};
    rocm::reverse_iterator<const point*> ri(arr + 3);
    EXPECT(ri->x == 5);
    ++ri;
    EXPECT(ri->x == 3);
    ++ri;
    EXPECT(ri->x == 1);
}

TEST_CASE(arrow_pointer_mutable)
{
    point arr[] = {{1, 2}, {3, 4}, {5, 6}};
    rocm::reverse_iterator<point*> ri(arr + 3);
    ri->x = 50;
    ri->y = 60;
    EXPECT(arr[2].x == 50);
    EXPECT(arr[2].y == 60);
}

TEST_CASE(arrow_class_iterator)
{
    point arr[] = {{1, 2}, {3, 4}, {5, 6}};
    auto ri     = rocm::make_reverse_iterator(wrapper_iterator<const point>{arr + 3});
    EXPECT(ri->x == 5);
    EXPECT(ri->y == 6);
}

TEST_CASE(arrow_class_iterator_traversal)
{
    point arr[] = {{1, 2}, {3, 4}, {5, 6}};
    auto ri     = rocm::make_reverse_iterator(wrapper_iterator<const point>{arr + 3});
    EXPECT(ri->x == 5);
    ++ri;
    EXPECT(ri->x == 3);
    ++ri;
    EXPECT(ri->x == 1);
}

TEST_CASE(arrow_class_iterator_mutable)
{
    point arr[] = {{1, 2}, {3, 4}, {5, 6}};
    auto ri     = rocm::make_reverse_iterator(wrapper_iterator<point>{arr + 2});
    ri->x       = 30;
    ri->y       = 40;
    EXPECT(arr[1].x == 30);
    EXPECT(arr[1].y == 40);
}

TEST_CASE(arrow_matches_dereference)
{
    point arr[] = {{1, 2}, {3, 4}, {5, 6}};
    auto ri     = rocm::make_reverse_iterator(wrapper_iterator<const point>{arr + 3});
    EXPECT(ri.operator->() == &(*ri));
}

// ---- Subscript ----

TEST_CASE(subscript)
{
    int arr[] = {10, 20, 30, 40, 50};
    rocm::reverse_iterator<const int*> ri(arr + 5);
    EXPECT(ri[0] == 50);
    EXPECT(ri[1] == 40);
    EXPECT(ri[2] == 30);
    EXPECT(ri[3] == 20);
    EXPECT(ri[4] == 10);
}

// ---- Pre-increment / Pre-decrement ----

TEST_CASE(pre_increment)
{
    int arr[] = {10, 20, 30};
    rocm::reverse_iterator<int*> ri(arr + 3);
    EXPECT(*ri == 30);
    ++ri;
    EXPECT(*ri == 20);
    ++ri;
    EXPECT(*ri == 10);
}

TEST_CASE(pre_decrement)
{
    int arr[] = {10, 20, 30};
    rocm::reverse_iterator<int*> ri(arr + 1);
    EXPECT(*ri == 10);
    --ri;
    EXPECT(*ri == 20);
    --ri;
    EXPECT(*ri == 30);
}

// ---- Post-increment / Post-decrement ----

TEST_CASE(post_increment)
{
    int arr[] = {10, 20, 30};
    rocm::reverse_iterator<int*> ri(arr + 3);
    auto old = ri++;
    EXPECT(*old == 30);
    EXPECT(*ri == 20);
}

TEST_CASE(post_decrement)
{
    int arr[] = {10, 20, 30};
    rocm::reverse_iterator<int*> ri(arr + 1);
    auto old = ri--;
    EXPECT(*old == 10);
    EXPECT(*ri == 20);
}

// ---- Compound assignment ----

TEST_CASE(plus_equal)
{
    int arr[] = {10, 20, 30, 40, 50};
    rocm::reverse_iterator<int*> ri(arr + 5);
    EXPECT(*ri == 50);
    ri += 3;
    EXPECT(*ri == 20);
}

TEST_CASE(minus_equal)
{
    int arr[] = {10, 20, 30, 40, 50};
    rocm::reverse_iterator<int*> ri(arr + 2);
    EXPECT(*ri == 20);
    ri -= 3;
    EXPECT(*ri == 50);
}

// ---- Arithmetic operators ----

TEST_CASE(add_offset)
{
    int arr[] = {10, 20, 30, 40, 50};
    rocm::reverse_iterator<const int*> ri(arr + 5);
    auto ri2 = ri + 2;
    EXPECT(*ri2 == 30);
}

TEST_CASE(offset_add)
{
    int arr[] = {10, 20, 30, 40, 50};
    rocm::reverse_iterator<const int*> ri(arr + 5);
    auto ri2 = 2 + ri;
    EXPECT(*ri2 == 30);
}

TEST_CASE(subtract_offset)
{
    int arr[] = {10, 20, 30, 40, 50};
    rocm::reverse_iterator<const int*> ri(arr + 2);
    auto ri2 = ri - 2;
    EXPECT(*ri2 == 40);
}

TEST_CASE(difference)
{
    int arr[] = {10, 20, 30, 40, 50};
    rocm::reverse_iterator<const int*> rbegin(arr + 5);
    rocm::reverse_iterator<const int*> rend(arr);
    EXPECT(rend - rbegin == 5);
    EXPECT(rbegin - rend == -5);
}

// ---- Comparison operators ----

TEST_CASE(equal)
{
    int arr[] = {1, 2, 3};
    rocm::reverse_iterator<const int*> a(arr + 3);
    rocm::reverse_iterator<const int*> b(arr + 3);
    rocm::reverse_iterator<const int*> c(arr + 2);
    EXPECT(a == b);
    EXPECT(not(a == c));
}

TEST_CASE(not_equal)
{
    int arr[] = {1, 2, 3};
    rocm::reverse_iterator<const int*> a(arr + 3);
    rocm::reverse_iterator<const int*> b(arr + 2);
    EXPECT(a != b);
    EXPECT(not(a != a));
}

TEST_CASE(less_than)
{
    int arr[] = {1, 2, 3};
    rocm::reverse_iterator<const int*> rbegin(arr + 3);
    rocm::reverse_iterator<const int*> rend(arr);
    EXPECT(rbegin < rend);
    EXPECT(not(rend < rbegin));
    EXPECT(not(rbegin < rbegin));
}

TEST_CASE(greater_than)
{
    int arr[] = {1, 2, 3};
    rocm::reverse_iterator<const int*> rbegin(arr + 3);
    rocm::reverse_iterator<const int*> rend(arr);
    EXPECT(rend > rbegin);
    EXPECT(not(rbegin > rend));
}

TEST_CASE(less_equal)
{
    int arr[] = {1, 2, 3};
    rocm::reverse_iterator<const int*> rbegin(arr + 3);
    rocm::reverse_iterator<const int*> rend(arr);
    EXPECT(rbegin <= rend);
    EXPECT(rbegin <= rbegin);
    EXPECT(not(rend <= rbegin));
}

TEST_CASE(greater_equal)
{
    int arr[] = {1, 2, 3};
    rocm::reverse_iterator<const int*> rbegin(arr + 3);
    rocm::reverse_iterator<const int*> rend(arr);
    EXPECT(rend >= rbegin);
    EXPECT(rend >= rend);
    EXPECT(not(rbegin >= rend));
}

// ---- make_reverse_iterator ----

TEST_CASE(make_reverse_iterator_test)
{
    int arr[] = {10, 20, 30};
    auto ri   = rocm::make_reverse_iterator(arr + 3);
    EXPECT(*ri == 30);
}

// ---- base() ----

TEST_CASE(base)
{
    int arr[] = {10, 20, 30};
    rocm::reverse_iterator<const int*> ri(arr + 2);
    EXPECT(ri.base() == arr + 2);
}

// ---- Full traversal ----

TEST_CASE(full_traversal)
{
    int arr[] = {1, 2, 3, 4, 5};
    rocm::reverse_iterator<int*> ri(arr + 5);
    int expected = 5;
    while(ri != rocm::reverse_iterator<int*>(arr))
    {
        EXPECT(*ri == expected);
        --expected;
        ++ri;
    }
    EXPECT(expected == 0);
}

// ---- constexpr evaluation via static_assert ----

TEST_CASE(constexpr_dereference)
{
    static constexpr int arr[] = {100, 200, 300};
    static_assert(*rocm::reverse_iterator<const int*>(arr + 3) == 300);
    static_assert(*rocm::reverse_iterator<const int*>(arr + 2) == 200);
    static_assert(*rocm::reverse_iterator<const int*>(arr + 1) == 100);
}

TEST_CASE(constexpr_subscript)
{
    static constexpr int arr[] = {10, 20, 30};
    static_assert(rocm::reverse_iterator<const int*>(arr + 3)[0] == 30);
    static_assert(rocm::reverse_iterator<const int*>(arr + 3)[1] == 20);
    static_assert(rocm::reverse_iterator<const int*>(arr + 3)[2] == 10);
}

TEST_CASE(constexpr_arithmetic)
{
    static constexpr int arr[] = {10, 20, 30, 40};
    static_assert(*(rocm::reverse_iterator<const int*>(arr + 4) + 2) == 20);
    static_assert(*(2 + rocm::reverse_iterator<const int*>(arr + 4)) == 20);
    static_assert(*(rocm::reverse_iterator<const int*>(arr + 1) - 1) == 20);
}

TEST_CASE(constexpr_arrow)
{
    static constexpr point arr[] = {{1, 2}, {3, 4}, {5, 6}};
    static_assert(rocm::reverse_iterator<const point*>(arr + 3)->x == 5);
    static_assert(rocm::reverse_iterator<const point*>(arr + 2)->y == 4);
    using iterator = wrapper_iterator<const point>;
    static_assert(rocm::make_reverse_iterator(iterator{arr + 3})->x == 5);
    static_assert(rocm::make_reverse_iterator(iterator{arr + 2})->y == 4);
}

TEST_CASE(constexpr_comparison)
{
    static constexpr int arr[] = {1, 2, 3};
    static_assert(rocm::reverse_iterator<const int*>(arr + 3) ==
                  rocm::reverse_iterator<const int*>(arr + 3));
    static_assert(rocm::reverse_iterator<const int*>(arr + 3) !=
                  rocm::reverse_iterator<const int*>(arr + 2));
    static_assert(rocm::reverse_iterator<const int*>(arr + 3) <
                  rocm::reverse_iterator<const int*>(arr));
}

// ---- Single element ----

TEST_CASE(single_element)
{
    int arr[] = {42};
    rocm::reverse_iterator<const int*> ri(arr + 1);
    EXPECT(*ri == 42);
    EXPECT(ri[0] == 42);
}
