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
#include <migraphx/span.hpp>
#include <migraphx/small_vector.hpp>
#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <iterator>
#include <list>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <test.hpp>

template <class Span, class = void>
struct has_writable_bytes : std::false_type
{
};

template <class Span>
struct has_writable_bytes<Span,
                          std::void_t<decltype(migraphx::as_writable_bytes(std::declval<Span>()))>>
    : std::true_type
{
};

template <class T, std::size_t N>
static bool views(const migraphx::span<T, N>& s, const std::vector<std::remove_cv_t<T>>& expected)
{
    return s.size() == expected.size() and std::equal(s.begin(), s.end(), expected.begin());
}

TEST_CASE(types)
{
    using dynamic = migraphx::span<int>;
    using fixed   = migraphx::span<const int, 3>;
    static_assert(dynamic::extent == migraphx::dynamic_extent);
    static_assert(fixed::extent == 3);
    static_assert(std::is_same<dynamic::element_type, int>{});
    static_assert(std::is_same<fixed::element_type, const int>{});
    static_assert(std::is_same<fixed::value_type, int>{});
    static_assert(std::is_same<dynamic::size_type, std::size_t>{});
    static_assert(std::is_same<dynamic::difference_type, std::ptrdiff_t>{});
    static_assert(std::is_same<dynamic::pointer, int*>{});
    static_assert(std::is_same<dynamic::const_pointer, const int*>{});
    static_assert(std::is_same<dynamic::reference, int&>{});
    static_assert(std::is_same<dynamic::const_reference, const int&>{});
    static_assert(std::is_same<dynamic::iterator, int*>{});
    static_assert(std::is_same<fixed::iterator, const int*>{});
    static_assert(std::is_same<dynamic::const_iterator, const int*>{});
    static_assert(std::is_same<dynamic::reverse_iterator, std::reverse_iterator<int*>>{});
    static_assert(sizeof(fixed) == sizeof(const int*));
    static_assert(sizeof(dynamic) == sizeof(int*) + sizeof(std::size_t));
    static_assert(std::is_trivially_copyable<dynamic>{});
    static_assert(std::is_trivially_copyable<fixed>{});
}

TEST_CASE(default_construct)
{
    static_assert(std::is_default_constructible<migraphx::span<int>>{});
    static_assert(std::is_default_constructible<migraphx::span<int, 0>>{});
    static_assert(not std::is_default_constructible<migraphx::span<int, 3>>{});
    migraphx::span<int> s;
    EXPECT(s.empty());
    EXPECT(s.size() == 0);
    EXPECT(s.data() == nullptr);
    EXPECT(s.begin() == s.end());
    migraphx::span<int, 0> z;
    EXPECT(z.empty());
    EXPECT(z.size() == 0);
    EXPECT(z.data() == nullptr);
}

TEST_CASE(construct_pointer_count)
{
    std::vector<int> v = {1, 2, 3, 4};
    migraphx::span<int> s(v.data(), 3);
    EXPECT(views(s, {1, 2, 3}));
    EXPECT(s.data() == v.data());
    migraphx::span<const int> cs(v.data(), v.size());
    EXPECT(views(cs, {1, 2, 3, 4}));
    migraphx::span<int, 4> f(v.data(), 4);
    EXPECT(views(f, {1, 2, 3, 4}));
    static_assert(not std::is_convertible<int*, migraphx::span<int>>{});
    static_assert(not std::is_constructible<migraphx::span<int>, const int*, std::size_t>{});
    static_assert(not std::is_constructible<migraphx::span<long>, int*, std::size_t>{});
    static_assert(
        not std::is_constructible<migraphx::span<int>, std::list<int>::iterator, std::size_t>{});
}

TEST_CASE(construct_pointer_pair)
{
    std::vector<int> v = {1, 2, 3, 4};
    migraphx::span<int> s(v.data() + 1, v.data() + 3);
    EXPECT(views(s, {2, 3}));
    migraphx::span<const int> cs(v.data(), v.data() + v.size());
    EXPECT(views(cs, {1, 2, 3, 4}));
    migraphx::span<int, 2> f(v.data(), v.data() + 2);
    EXPECT(views(f, {1, 2}));
    migraphx::span<int> e(v.data(), v.data());
    EXPECT(e.empty());
}

TEST_CASE(construct_c_array)
{
    int arr[]        = {1, 2, 3};
    const int carr[] = {4, 5};
    migraphx::span<int> s(arr);
    EXPECT(views(s, {1, 2, 3}));
    EXPECT(s.data() == arr);
    migraphx::span<int, 3> f(arr);
    EXPECT(views(f, {1, 2, 3}));
    migraphx::span<const int> cs(arr);
    EXPECT(views(cs, {1, 2, 3}));
    migraphx::span<const int, 2> cf(carr);
    EXPECT(views(cf, {4, 5}));
    static_assert(std::is_convertible<int (&)[3], migraphx::span<int, 3>>{});
    static_assert(not std::is_constructible<migraphx::span<int, 2>, int (&)[3]>{});
    static_assert(not std::is_constructible<migraphx::span<int>, const int (&)[2]>{});
}

TEST_CASE(construct_std_array)
{
    std::array<int, 3> arr        = {{1, 2, 3}};
    const std::array<int, 2> carr = {{4, 5}};
    migraphx::span<int> s(arr);
    EXPECT(views(s, {1, 2, 3}));
    EXPECT(s.data() == arr.data());
    migraphx::span<int, 3> f(arr);
    EXPECT(views(f, {1, 2, 3}));
    migraphx::span<const int, 3> cf(arr);
    EXPECT(views(cf, {1, 2, 3}));
    migraphx::span<const int> cs(carr);
    EXPECT(views(cs, {4, 5}));
    migraphx::span<const int, 2> cf2(carr);
    EXPECT(views(cf2, {4, 5}));
    static_assert(std::is_convertible<std::array<int, 3>&, migraphx::span<int, 3>>{});
    static_assert(not std::is_constructible<migraphx::span<int, 2>, std::array<int, 3>&>{});
    static_assert(not std::is_constructible<migraphx::span<int>, const std::array<int, 2>&>{});
    static_assert(not std::is_constructible<migraphx::span<long>, std::array<int, 2>&>{});
}

TEST_CASE(construct_range)
{
    std::vector<int> v        = {1, 2, 3};
    const std::vector<int> cv = {4, 5};
    migraphx::span<int> s(v);
    EXPECT(views(s, {1, 2, 3}));
    EXPECT(s.data() == v.data());
    migraphx::span<const int> cs(v);
    EXPECT(views(cs, {1, 2, 3}));
    migraphx::span<const int> cs2(cv);
    EXPECT(views(cs2, {4, 5}));
    migraphx::span<int, 3> f(v);
    EXPECT(views(f, {1, 2, 3}));
    static_assert(std::is_convertible<std::vector<int>&, migraphx::span<int>>{});
    static_assert(std::is_convertible<std::vector<int>&, migraphx::span<const int>>{});
    static_assert(not std::is_convertible<std::vector<int>&, migraphx::span<int, 3>>{});
    static_assert(std::is_constructible<migraphx::span<int, 3>, std::vector<int>&>{});
    static_assert(not std::is_constructible<migraphx::span<int>, const std::vector<int>&>{});
    static_assert(not std::is_constructible<migraphx::span<int>, std::list<int>&>{});
    static_assert(not std::is_constructible<migraphx::span<bool>, std::vector<bool>&>{});
    static_assert(not std::is_constructible<migraphx::span<long>, std::vector<int>&>{});
}

TEST_CASE(construct_temporary_range)
{
    // A temporary may only be viewed through a span of const elements
    static_assert(std::is_constructible<migraphx::span<const int>, std::vector<int>>{});
    static_assert(not std::is_constructible<migraphx::span<int>, std::vector<int>>{});
}

TEST_CASE(construct_string)
{
    std::string str = "hello";
    migraphx::span<char> s(str);
    EXPECT(s.size() == 5);
    s[0] = 'j';
    EXPECT(str == "jello");
    migraphx::span<const char> cs(str);
    EXPECT(cs.front() == 'j');
    EXPECT(cs.back() == 'o');
}

TEST_CASE(construct_small_vector)
{
    migraphx::small_vector<int, 4> sv = {1, 2, 3};
    migraphx::span<int> s(sv);
    EXPECT(views(s, {1, 2, 3}));
    EXPECT(s.data() == sv.data());
    migraphx::span deduced(sv);
    static_assert(std::is_same<decltype(deduced), migraphx::span<int>>{});
    EXPECT(views(deduced, {1, 2, 3}));
}

TEST_CASE(construct_initializer_list)
{
    // The array behind a braced list only lives for the full expression, so a span of one is
    // held through a named initializer_list here
    std::initializer_list<int> il = {1, 2, 3};
    migraphx::span<const int> s   = il;
    EXPECT(views(s, {1, 2, 3}));
    std::initializer_list<int> il2 = {4, 5};
    migraphx::span<const int, 2> f(il2);
    EXPECT(views(f, {4, 5}));
    migraphx::span<const int> e = {};
    EXPECT(e.empty());
    static_assert(std::is_convertible<std::initializer_list<int>, migraphx::span<const int>>{});
    static_assert(
        not std::is_convertible<std::initializer_list<int>, migraphx::span<const int, 2>>{});
    static_assert(
        std::is_constructible<migraphx::span<const int, 2>, std::initializer_list<int>>{});
    static_assert(not std::is_constructible<migraphx::span<int>, std::initializer_list<int>>{});
}

static int sum_span(migraphx::span<const int> s) { return std::accumulate(s.begin(), s.end(), 0); }

TEST_CASE(pass_braced_list)
{
    std::vector<int> v   = {1, 2};
    std::array<int, 2> a = {{3, 4}};
    EXPECT(sum_span({5, 6, 7}) == 18);
    EXPECT(sum_span(v) == 3);
    EXPECT(sum_span(a) == 7);
    EXPECT(sum_span({}) == 0);
    EXPECT(sum_span({v.front()}) == 1);
}

TEST_CASE(construct_from_span)
{
    std::vector<int> v = {1, 2, 3};
    migraphx::span<int, 3> f(v);
    migraphx::span<int> d = f;
    EXPECT(views(d, {1, 2, 3}));
    migraphx::span<const int> cd = f;
    EXPECT(views(cd, {1, 2, 3}));
    migraphx::span<const int, 3> cf = f;
    EXPECT(views(cf, {1, 2, 3}));
    migraphx::span<int, 3> back(d);
    EXPECT(views(back, {1, 2, 3}));
    migraphx::span<int> copy(d);
    EXPECT(views(copy, {1, 2, 3}));
    static_assert(std::is_convertible<migraphx::span<int, 3>, migraphx::span<int>>{});
    static_assert(std::is_convertible<migraphx::span<int>, migraphx::span<const int>>{});
    static_assert(not std::is_convertible<migraphx::span<int>, migraphx::span<int, 3>>{});
    static_assert(std::is_constructible<migraphx::span<int, 3>, migraphx::span<int>>{});
    static_assert(not std::is_constructible<migraphx::span<int>, migraphx::span<const int>>{});
    static_assert(not std::is_constructible<migraphx::span<int, 2>, migraphx::span<int, 3>>{});
    static_assert(not std::is_constructible<migraphx::span<long>, migraphx::span<int>>{});
}

TEST_CASE(copy_assign)
{
    std::vector<int> v = {1, 2, 3};
    std::vector<int> w = {4, 5};
    migraphx::span<int> s(v);
    migraphx::span<int> t(w);
    s = t;
    EXPECT(views(s, {4, 5}));
    EXPECT(s.data() == w.data());
    migraphx::span<int, 2> f(w);
    migraphx::span<int, 2> g(v.data(), 2);
    f = g;
    EXPECT(views(f, {1, 2}));
}

TEST_CASE(element_access)
{
    std::vector<int> v = {1, 2, 3};
    migraphx::span<int> s(v);
    EXPECT(s[0] == 1);
    EXPECT(s[2] == 3);
    EXPECT(s.at(1) == 2);
    EXPECT(s.front() == 1);
    EXPECT(s.back() == 3);
    EXPECT(s.data() == v.data());
    s[1]      = 20;
    s.front() = 10;
    s.back()  = 30;
    s.at(1) += 2;
    EXPECT(v == std::vector<int>{10, 22, 30});
    EXPECT(test::throws<std::out_of_range>([&] { s.at(3); }));
    EXPECT(test::throws<std::out_of_range>([&] { s.at(100); }));
    migraphx::span<int> e;
    EXPECT(test::throws<std::out_of_range>([&] { e.at(0); }));
}

TEST_CASE(const_span_writes_through)
{
    std::vector<int> v = {1, 2, 3};
    const migraphx::span<int> s(v);
    s[0]      = 7;
    s.front() = 8;
    *s.begin() += 1;
    EXPECT(v == std::vector<int>{9, 2, 3});
    static_assert(std::is_same<decltype(s[0]), int&>{});
}

TEST_CASE(iterators)
{
    std::vector<int> v = {1, 2, 3};
    migraphx::span<int> s(v);
    EXPECT(s.begin() == v.data());
    EXPECT(s.end() == v.data() + 3);
    EXPECT(std::distance(s.begin(), s.end()) == 3);
    EXPECT(s.cbegin() == s.begin());
    EXPECT(s.cend() == s.end());
    EXPECT(*s.rbegin() == 3);
    EXPECT(*std::prev(s.rend()) == 1);
    EXPECT(*s.crbegin() == 3);
    EXPECT(*std::prev(s.crend()) == 1);
    EXPECT(std::vector<int>(s.rbegin(), s.rend()) == std::vector<int>{3, 2, 1});
    static_assert(std::is_same<decltype(s.cbegin()), const int*>{});
    static_assert(std::is_same<decltype(s.crbegin()), std::reverse_iterator<const int*>>{});
    int total = 0;
    std::for_each(s.begin(), s.end(), [&](int x) { total += x; });
    EXPECT(total == 6);
}

TEST_CASE(size_bytes)
{
    std::vector<int> v = {1, 2, 3};
    migraphx::span<int> s(v);
    EXPECT(s.size() == 3);
    EXPECT(s.size_bytes() == 3 * sizeof(int));
    EXPECT(not s.empty());
    migraphx::span<int, 3> f(v);
    EXPECT(f.size() == 3);
    EXPECT(f.size_bytes() == 3 * sizeof(int));
    std::vector<double> d(5);
    EXPECT(migraphx::span<double>(d).size_bytes() == 5 * sizeof(double));
    EXPECT(migraphx::span<double>(d.data(), 0).empty());
}

TEST_CASE(first_last_fixed_count)
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    migraphx::span<int> s(v);
    migraphx::span<int, 5> f(v);
    auto s2 = s.first<2>();
    static_assert(std::is_same<decltype(s2), migraphx::span<int, 2>>{});
    EXPECT(views(s2, {1, 2}));
    auto f2 = f.first<2>();
    static_assert(std::is_same<decltype(f2), migraphx::span<int, 2>>{});
    EXPECT(views(f2, {1, 2}));
    auto s3 = s.last<3>();
    static_assert(std::is_same<decltype(s3), migraphx::span<int, 3>>{});
    EXPECT(views(s3, {3, 4, 5}));
    auto f3 = f.last<3>();
    static_assert(std::is_same<decltype(f3), migraphx::span<int, 3>>{});
    EXPECT(views(f3, {3, 4, 5}));
    EXPECT(f.first<0>().empty());
    EXPECT(f.last<0>().empty());
    EXPECT(views(f.first<5>(), {1, 2, 3, 4, 5}));
    EXPECT(views(f.last<5>(), {1, 2, 3, 4, 5}));
}

TEST_CASE(first_last_dynamic_count)
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    migraphx::span<int> s(v);
    migraphx::span<int, 5> f(v);
    auto s2 = s.first(2);
    static_assert(std::is_same<decltype(s2), migraphx::span<int>>{});
    EXPECT(views(s2, {1, 2}));
    auto f2 = f.first(2);
    static_assert(std::is_same<decltype(f2), migraphx::span<int>>{});
    EXPECT(views(f2, {1, 2}));
    EXPECT(views(s.last(3), {3, 4, 5}));
    EXPECT(views(f.last(3), {3, 4, 5}));
    EXPECT(s.first(0).empty());
    EXPECT(s.last(0).empty());
    EXPECT(views(s.first(s.size()), {1, 2, 3, 4, 5}));
    EXPECT(views(s.last(s.size()), {1, 2, 3, 4, 5}));
    EXPECT(s.first(2).data() == v.data());
    EXPECT(s.last(2).data() == v.data() + 3);
}

TEST_CASE(subspan_fixed)
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    migraphx::span<int> s(v);
    migraphx::span<int, 5> f(v);
    auto f1 = f.subspan<1>();
    static_assert(std::is_same<decltype(f1), migraphx::span<int, 4>>{});
    EXPECT(views(f1, {2, 3, 4, 5}));
    auto f12 = f.subspan<1, 2>();
    static_assert(std::is_same<decltype(f12), migraphx::span<int, 2>>{});
    EXPECT(views(f12, {2, 3}));
    auto s1 = s.subspan<1>();
    static_assert(std::is_same<decltype(s1), migraphx::span<int>>{});
    EXPECT(views(s1, {2, 3, 4, 5}));
    auto s12 = s.subspan<1, 2>();
    static_assert(std::is_same<decltype(s12), migraphx::span<int, 2>>{});
    EXPECT(views(s12, {2, 3}));
    EXPECT(f.subspan<5>().empty());
    static_assert(std::is_same<decltype(f.subspan<5>()), migraphx::span<int, 0>>{});
    EXPECT(views(f.subspan<0>(), {1, 2, 3, 4, 5}));
    EXPECT(views(f.subspan<3, 2>(), {4, 5}));
}

TEST_CASE(subspan_dynamic)
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    migraphx::span<int> s(v);
    migraphx::span<int, 5> f(v);
    auto s1 = s.subspan(1);
    static_assert(std::is_same<decltype(s1), migraphx::span<int>>{});
    EXPECT(views(s1, {2, 3, 4, 5}));
    EXPECT(views(s.subspan(1, 2), {2, 3}));
    auto f1 = f.subspan(1);
    static_assert(std::is_same<decltype(f1), migraphx::span<int>>{});
    EXPECT(views(f1, {2, 3, 4, 5}));
    EXPECT(views(f.subspan(1, 2), {2, 3}));
    EXPECT(s.subspan(5).empty());
    EXPECT(s.subspan(2, 0).empty());
    EXPECT(views(s.subspan(0), {1, 2, 3, 4, 5}));
    EXPECT(views(s.subspan(3, 2), {4, 5}));
    EXPECT(views(s.subspan(3, migraphx::dynamic_extent), {4, 5}));
    EXPECT(s.subspan(1, 2).data() == v.data() + 1);
}

TEST_CASE(subviews_write_through)
{
    std::vector<int> v = {1, 2, 3, 4, 5};
    migraphx::span<int> s(v);
    std::fill(s.subspan(1, 3).begin(), s.subspan(1, 3).end(), 0);
    EXPECT(v == std::vector<int>{1, 0, 0, 0, 5});
    s.first(1)[0] = 9;
    s.last(1)[0]  = 8;
    EXPECT(v == std::vector<int>{9, 0, 0, 0, 8});
}

TEST_CASE(bytes)
{
    std::array<int, 2> arr = {{1, 2}};
    migraphx::span<int, 2> f(arr);
    migraphx::span<int> s(arr);
    auto fb = migraphx::as_bytes(f);
    static_assert(
        std::is_same<decltype(fb), migraphx::span<const migraphx::byte, 2 * sizeof(int)>>{});
    EXPECT(fb.size() == 2 * sizeof(int));
    EXPECT(static_cast<const void*>(fb.data()) == static_cast<const void*>(arr.data()));
    auto sb = migraphx::as_bytes(s);
    static_assert(std::is_same<decltype(sb), migraphx::span<const migraphx::byte>>{});
    EXPECT(sb.size() == s.size_bytes());
    auto wb = migraphx::as_writable_bytes(s);
    static_assert(std::is_same<decltype(wb), migraphx::span<migraphx::byte>>{});
    EXPECT(wb.size() == s.size_bytes());
    std::fill(wb.begin(), wb.end(), migraphx::byte{0});
    EXPECT(arr == std::array<int, 2>{{0, 0}});
    auto wfb = migraphx::as_writable_bytes(f);
    static_assert(std::is_same<decltype(wfb), migraphx::span<migraphx::byte, 2 * sizeof(int)>>{});
    static_assert(has_writable_bytes<migraphx::span<int>>{});
    static_assert(not has_writable_bytes<migraphx::span<const int>>{});
    migraphx::span<int, 0> z;
    static_assert(
        std::is_same<decltype(migraphx::as_bytes(z)), migraphx::span<const migraphx::byte, 0>>{});
    EXPECT(migraphx::as_bytes(z).empty());
}

TEST_CASE(deduction)
{
    int arr[]                     = {1, 2, 3};
    std::array<int, 2> sarr       = {{4, 5}};
    const std::array<int, 2> carr = {{6, 7}};
    std::vector<int> v            = {8, 9};
    const std::vector<int> cv     = {10};
    std::string str               = "ab";

    migraphx::span s1(arr);
    static_assert(std::is_same<decltype(s1), migraphx::span<int, 3>>{});
    EXPECT(views(s1, {1, 2, 3}));
    migraphx::span s2(sarr);
    static_assert(std::is_same<decltype(s2), migraphx::span<int, 2>>{});
    migraphx::span s3(carr);
    static_assert(std::is_same<decltype(s3), migraphx::span<const int, 2>>{});
    EXPECT(views(s3, {6, 7}));
    migraphx::span s4(v);
    static_assert(std::is_same<decltype(s4), migraphx::span<int>>{});
    EXPECT(views(s4, {8, 9}));
    migraphx::span s5(cv);
    static_assert(std::is_same<decltype(s5), migraphx::span<const int>>{});
    EXPECT(views(s5, {10}));
    migraphx::span s6(v.data(), v.size());
    static_assert(std::is_same<decltype(s6), migraphx::span<int>>{});
    migraphx::span s7(v.data(), v.data() + 1);
    static_assert(std::is_same<decltype(s7), migraphx::span<int>>{});
    EXPECT(views(s7, {8}));
    migraphx::span s8(cv.data(), 1);
    static_assert(std::is_same<decltype(s8), migraphx::span<const int>>{});
    migraphx::span s9(s1);
    static_assert(std::is_same<decltype(s9), migraphx::span<int, 3>>{});
    migraphx::span s10(str);
    static_assert(std::is_same<decltype(s10), migraphx::span<char>>{});
    EXPECT(s10.size() == 2);
}

TEST_CASE(constexpr_usage)
{
    static constexpr std::array<int, 4> arr = {{1, 2, 3, 4}};
    constexpr migraphx::span<const int, 4> f(arr);
    static_assert(f.size() == 4);
    static_assert(f.size_bytes() == 4 * sizeof(int));
    static_assert(not f.empty());
    static_assert(f[1] == 2);
    static_assert(f.front() == 1);
    static_assert(f.back() == 4);
    static_assert(f.data() == arr.data());
    static_assert(*f.begin() == 1);
    static_assert(f.end() - f.begin() == 4);
    static_assert(f.first<2>()[1] == 2);
    static_assert(f.last<2>()[0] == 3);
    static_assert(f.subspan<1, 2>()[0] == 2);
    static_assert(f.subspan<2>().size() == 2);
    static_assert(f.first(3).size() == 3);
    static_assert(f.last(1)[0] == 4);
    static_assert(f.subspan(1, 2)[1] == 3);
    constexpr migraphx::span<const int> d = f;
    static_assert(d.size() == 4);
    static_assert(d[3] == 4);
    constexpr migraphx::span<const int> e;
    static_assert(e.empty());
    static_assert(e.data() == nullptr);
}

TEST_CASE(algorithms)
{
    std::vector<int> v = {3, 1, 2};
    migraphx::span<int> s(v);
    std::sort(s.begin(), s.end());
    EXPECT(v == std::vector<int>{1, 2, 3});
    EXPECT(std::accumulate(s.begin(), s.end(), 0) == 6);
    EXPECT(*std::find(s.begin(), s.end(), 2) == 2);
    EXPECT(std::find(s.begin(), s.end(), 5) == s.end());
    std::transform(s.begin(), s.end(), s.begin(), [](int x) { return x * 10; });
    EXPECT(v == std::vector<int>{10, 20, 30});
    std::vector<int> reversed;
    std::reverse_copy(s.begin(), s.end(), std::back_inserter(reversed));
    EXPECT(reversed == std::vector<int>{30, 20, 10});
    EXPECT(std::equal(s.begin(), s.end(), v.begin(), v.end()));
}

TEST_CASE(class_elements)
{
    std::vector<std::string> v = {"a", "bb", "ccc"};
    migraphx::span<std::string> s(v);
    EXPECT(s.front().size() == 1);
    EXPECT(s.back() == "ccc");
    s[1] += "!";
    EXPECT(v[1] == "bb!");
    EXPECT(s.begin()->size() == 1);
    migraphx::span<const std::string> cs(v);
    EXPECT(cs[2].size() == 3);
}

TEST_CASE(span_of_spans)
{
    std::vector<int> a                     = {1, 2};
    std::vector<int> b                     = {3};
    std::vector<migraphx::span<int>> parts = {migraphx::span<int>(a), migraphx::span<int>(b)};
    migraphx::span<migraphx::span<int>> s(parts);
    EXPECT(s.size() == 2);
    EXPECT(s[0].size() == 2);
    EXPECT(s[1][0] == 3);
    s[1][0] = 4;
    EXPECT(b[0] == 4);
}

TEST_CASE(zero_extent)
{
    migraphx::span<int, 0> z;
    EXPECT(z.empty());
    EXPECT(z.size() == 0);
    EXPECT(z.size_bytes() == 0);
    EXPECT(z.begin() == z.end());
    EXPECT(z.first<0>().empty());
    EXPECT(z.last<0>().empty());
    EXPECT(z.subspan<0>().empty());
    EXPECT(z.first(0).empty());
    EXPECT(z.subspan(0).empty());
    std::vector<int> v = {1};
    migraphx::span<int, 0> from_data(v.data(), 0);
    EXPECT(from_data.empty());
    EXPECT(from_data.data() == v.data());
    migraphx::span<int> d = z;
    EXPECT(d.empty());
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
