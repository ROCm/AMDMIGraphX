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
#include <migraphx/picked_variant.hpp>
#include <test.hpp>

#include <string>
#include <type_traits>
#include <variant>

struct copy_picker
{
    template <class T>
    static decltype(auto) apply(T&& x)
    {
        return std::forward<T>(x);
    }
};

using pv_t = migraphx::picked_variant<copy_picker, int, long, std::string>;

struct always_long
{
    template <class T>
    static long apply(T&&)
    {
        return 999L;
    }
};

using long_pv = migraphx::picked_variant<always_long, int, long, std::string>;

TEST_CASE(default_ctor)
{
    pv_t v;
    EXPECT(v.index() == 0);
    EXPECT(std::holds_alternative<int>(v));
    EXPECT(std::get<int>(v) == 0);
}

TEST_CASE(in_place_type_ctor_int)
{
    pv_t v(std::in_place_type<int>, 42);
    EXPECT(v.index() == 0);
    EXPECT(std::holds_alternative<int>(v));
    EXPECT(std::get<int>(v) == 42);
}

TEST_CASE(in_place_type_ctor_long)
{
    pv_t v(std::in_place_type<long>, 7L);
    EXPECT(v.index() == 1);
    EXPECT(std::holds_alternative<long>(v));
    EXPECT(std::get<long>(v) == 7L);
}

TEST_CASE(in_place_type_ctor_string)
{
    pv_t v(std::in_place_type<std::string>, "hello");
    EXPECT(v.index() == 2);
    EXPECT(std::holds_alternative<std::string>(v));
    EXPECT(std::get<std::string>(v) == "hello");
}

TEST_CASE(in_place_index_ctor)
{
    pv_t v(std::in_place_index<2>, "world");
    EXPECT(v.index() == 2);
    EXPECT(std::holds_alternative<std::string>(v));
    EXPECT(std::get<std::string>(v) == "world");
}

TEST_CASE(holds_alternative_negative)
{
    pv_t v(std::in_place_type<int>, 42);
    EXPECT(std::holds_alternative<int>(v));
    EXPECT(not std::holds_alternative<long>(v));
    EXPECT(not std::holds_alternative<std::string>(v));
}

TEST_CASE(get_by_type)
{
    pv_t v(std::in_place_type<int>, 123);
    EXPECT(std::get<int>(v) == 123);
    EXPECT(test::throws([&] { (void)std::get<long>(v); }));
    EXPECT(test::throws([&] { (void)std::get<std::string>(v); }));
}

TEST_CASE(get_by_index)
{
    pv_t v(std::in_place_index<1>, 99L);
    EXPECT(std::get<1>(v) == 99L);
    EXPECT(test::throws([&] { (void)std::get<0>(v); }));
}

TEST_CASE(get_if_pointer)
{
    pv_t v(std::in_place_type<int>, 7);
    auto* ip = std::get_if<int>(&v);
    EXPECT(ip != nullptr);
    EXPECT(*ip == 7);
    auto* lp = std::get_if<long>(&v);
    EXPECT(lp == nullptr);
}

TEST_CASE(visit_returns_value)
{
    pv_t v(std::in_place_type<long>, 5L);
    auto doubled = std::visit(
        [](auto x) -> long {
            if constexpr(std::is_same_v<decltype(x), std::string>)
                return x.size();
            else
                return x * 2;
        },
        v);
    EXPECT(doubled == 10L);
}

TEST_CASE(visit_mutates_value)
{
    pv_t v(std::in_place_type<int>, 1);
    std::visit(
        [](auto& x) {
            if constexpr(std::is_arithmetic_v<std::decay_t<decltype(x)>>)
                x += 4;
        },
        v);
    EXPECT(std::get<int>(v) == 5);
}

TEST_CASE(copy_ctor)
{
    pv_t v1(std::in_place_type<std::string>, "copy");
    pv_t v2 = v1; // NOLINT(performance-unnecessary-copy-initialization)
    EXPECT(std::holds_alternative<std::string>(v2));
    EXPECT(std::get<std::string>(v2) == "copy");
}

TEST_CASE(move_ctor)
{
    pv_t v1(std::in_place_type<std::string>, "move");
    pv_t v2 = std::move(v1);
    EXPECT(std::holds_alternative<std::string>(v2));
    EXPECT(std::get<std::string>(v2) == "move");
}

TEST_CASE(emplace_changes_alternative)
{
    pv_t v(std::in_place_type<int>, 1);
    v.emplace<std::string>("now-string");
    EXPECT(std::holds_alternative<std::string>(v));
    EXPECT(std::get<std::string>(v) == "now-string");
    v.emplace<int>(42);
    EXPECT(std::holds_alternative<int>(v));
    EXPECT(std::get<int>(v) == 42);
}

TEST_CASE(equality)
{
    pv_t a(std::in_place_type<int>, 42);
    pv_t b(std::in_place_type<int>, 42);
    pv_t c(std::in_place_type<long>, 42L);
    EXPECT(a == b);
    EXPECT(a != c);
}

TEST_CASE(equality_same_alternative_different_value)
{
    pv_t a(std::in_place_type<int>, 1);
    pv_t b(std::in_place_type<int>, 2);
    EXPECT(not(a == b));
    EXPECT(a != b);
}

TEST_CASE(less_than_same_alternative)
{
    pv_t a(std::in_place_type<int>, 1);
    pv_t b(std::in_place_type<int>, 2);
    EXPECT(a < b);
    EXPECT(a <= b);
    EXPECT(b > a);
    EXPECT(b >= a);
    EXPECT(not(a > b));
    EXPECT(not(b < a));
}

TEST_CASE(less_than_different_alternative)
{
    pv_t a(std::in_place_type<int>, 100);
    pv_t b(std::in_place_type<long>, 0L);
    EXPECT(a < b);
    EXPECT(a <= b);
    EXPECT(b > a);
    EXPECT(b >= a);
}

TEST_CASE(less_equal_when_equal)
{
    pv_t a(std::in_place_type<long>, 7L);
    pv_t b(std::in_place_type<long>, 7L);
    EXPECT(a <= b);
    EXPECT(a >= b);
    EXPECT(not(a < b));
    EXPECT(not(a > b));
}

TEST_CASE(comparison_strings)
{
    pv_t a(std::in_place_type<std::string>, "abc");
    pv_t b(std::in_place_type<std::string>, "abd");
    EXPECT(a < b);
    EXPECT(a != b);
    EXPECT(b > a);
}

TEST_CASE(swap_alternatives)
{
    pv_t a(std::in_place_type<int>, 1);
    pv_t b(std::in_place_type<std::string>, "two");
    a.swap(b);
    EXPECT(std::holds_alternative<std::string>(a));
    EXPECT(std::get<std::string>(a) == "two");
    EXPECT(std::holds_alternative<int>(b));
    EXPECT(std::get<int>(b) == 1);
}

TEST_CASE(value_ctor_int_invokes_picker)
{
    pv_t v(42);
    EXPECT(std::holds_alternative<int>(v));
    EXPECT(std::get<int>(v) == 42);
}

TEST_CASE(value_ctor_long_invokes_picker)
{
    pv_t v(42L);
    EXPECT(std::holds_alternative<long>(v));
    EXPECT(std::get<long>(v) == 42L);
}

TEST_CASE(value_ctor_string_invokes_picker)
{
    pv_t v(std::string{"hi"});
    EXPECT(std::holds_alternative<std::string>(v));
    EXPECT(std::get<std::string>(v) == "hi");
}

TEST_CASE(value_ctor_const_char_ptr_invokes_picker)
{
    pv_t v("hi");
    EXPECT(std::holds_alternative<std::string>(v));
    EXPECT(std::get<std::string>(v) == "hi");
}

TEST_CASE(picker_can_redirect_alternative)
{
    long_pv v(42);
    EXPECT(std::holds_alternative<long>(v));
    EXPECT(std::get<long>(v) == 999L);
}

TEST_CASE(picker_redirects_string_to_long)
{
    long_pv v(std::string{"ignored"});
    EXPECT(std::holds_alternative<long>(v));
    EXPECT(std::get<long>(v) == 999L);
}

TEST_CASE(copy_does_not_invoke_picker)
{
    long_pv v1(std::in_place_type<int>, 7);
    long_pv v2 = v1; // NOLINT(performance-unnecessary-copy-initialization)
    EXPECT(std::holds_alternative<int>(v2));
    EXPECT(std::get<int>(v2) == 7);
}

struct route_by_type
{
    static int apply(int x) { return x + 1; }
    static long apply(long x) { return x + 100; }
    static std::string apply(const char* s) { return std::string{"got:"} + s; }
    static std::string apply(std::string s) { return "str:" + s; }
};

using route_pv = migraphx::picked_variant<route_by_type, int, long, std::string>;

TEST_CASE(picker_overload_int)
{
    route_pv v(5);
    EXPECT(std::holds_alternative<int>(v));
    EXPECT(std::get<int>(v) == 6);
}

TEST_CASE(picker_overload_long)
{
    route_pv v(5L);
    EXPECT(std::holds_alternative<long>(v));
    EXPECT(std::get<long>(v) == 105L);
}

TEST_CASE(picker_overload_const_char)
{
    route_pv v("hi");
    EXPECT(std::holds_alternative<std::string>(v));
    EXPECT(std::get<std::string>(v) == "got:hi");
}

TEST_CASE(picker_overload_string)
{
    route_pv v(std::string{"hi"});
    EXPECT(std::holds_alternative<std::string>(v));
    EXPECT(std::get<std::string>(v) == "str:hi");
}

struct count_picker
{
    static int& counter()
    {
        static int n = 0;
        return n;
    }
    template <class T>
    static decltype(auto) apply(T&& x)
    {
        ++counter();
        return std::forward<T>(x);
    }
};

using counted_pv = migraphx::picked_variant<count_picker, int, long, std::string>;

TEST_CASE(picker_invoked_exactly_once_per_value_ctor)
{
    count_picker::counter() = 0;
    counted_pv a(1);
    counted_pv b(2L);
    counted_pv c(std::string{"x"});
    EXPECT(count_picker::counter() == 3);
}

TEST_CASE(picker_not_invoked_for_default_ctor)
{
    count_picker::counter() = 0;
    counted_pv v;
    (void)v;
    EXPECT(count_picker::counter() == 0);
}

TEST_CASE(picker_not_invoked_for_in_place_ctor)
{
    count_picker::counter() = 0;
    counted_pv v(std::in_place_type<long>, 5L);
    (void)v;
    EXPECT(count_picker::counter() == 0);
}

TEST_CASE(picker_not_invoked_for_copy)
{
    counted_pv source(std::in_place_type<int>, 9);
    count_picker::counter()  = 0;
    counted_pv copied        = source; // NOLINT(performance-unnecessary-copy-initialization)
    counted_pv moved         = std::move(source);
    EXPECT(count_picker::counter() == 0);
    EXPECT(std::holds_alternative<int>(copied));
    EXPECT(std::holds_alternative<int>(moved));
}

struct lvalue_or_rvalue_picker
{
    static std::string apply(int&) { return "lvalue"; }
    static std::string apply(const int&) { return "const-lvalue"; }
    static std::string apply(int&&) { return "rvalue"; }
};

using vc_pv = migraphx::picked_variant<lvalue_or_rvalue_picker, int, long, std::string>;

TEST_CASE(picker_receives_rvalue)
{
    vc_pv v(42);
    EXPECT(std::get<std::string>(v) == "rvalue");
}

TEST_CASE(picker_receives_lvalue)
{
    int x = 42;
    vc_pv v(x);
    EXPECT(std::get<std::string>(v) == "lvalue");
}

TEST_CASE(picker_receives_const_lvalue)
{
    const int x = 42;
    vc_pv v(x);
    EXPECT(std::get<std::string>(v) == "const-lvalue");
}

TEST_CASE(is_derived_from_variant)
{
    static_assert(std::is_base_of<std::variant<int, long, std::string>, pv_t>{},
                  "picked_variant must derive from std::variant");
    pv_t v(std::in_place_type<int>, 42);
    std::variant<int, long, std::string>& base_ref = v;
    EXPECT(std::get<int>(base_ref) == 42);
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
