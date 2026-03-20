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

#include <migraphx/utility_operators.hpp>
#include <test.hpp>

#ifdef CPPCHECK
#define EXPECT_TOTALLY_ORDERED(...)
#else
// NOLINTNEXTLINE
#define EXPECT_TOTALLY_ORDERED_IMPL(x, y)     \
    EXPECT((x <= y) or (x >= y));             \
    EXPECT((x < y) or (x > y) or (x == y));   \
    EXPECT(((x < y) or (x > y)) == (x != y)); \
    EXPECT((x < y) == (y > x));               \
    EXPECT((x <= y) == (y >= x));             \
    EXPECT((x < y) != (x >= y));              \
    EXPECT((x > y) != (x <= y));              \
    EXPECT((x == y) != (x != y))

// NOLINTNEXTLINE
#define EXPECT_TOTALLY_ORDERED(x, y)   \
    EXPECT_TOTALLY_ORDERED_IMPL(x, y); \
    EXPECT_TOTALLY_ORDERED_IMPL(y, x)
#endif

struct custom_compare_any : migraphx::totally_ordered<custom_compare_any>
{
    int x;

    constexpr custom_compare_any(int px) : x(px) {}

    constexpr bool operator==(const custom_compare_any& rhs) const { return x == rhs.x; }

    template <class T>
    constexpr auto operator==(const T& rhs) const -> decltype(std::declval<int>() == rhs)
    {
        return x == rhs;
    }

    constexpr bool operator<(const custom_compare_any& rhs) const { return x < rhs.x; }

    template <class T>
    constexpr auto operator<(const T& rhs) const -> decltype(std::declval<int>() < rhs)
    {
        return x < rhs;
    }

    template <class T>
    constexpr auto operator>(const T& rhs) const -> decltype(std::declval<int>() > rhs)
    {
        return x > rhs;
    }

    template <class Stream>
    friend Stream& operator<<(Stream& os, const custom_compare_any& self)
    {
        return os << self.x;
    }
};

TEST_CASE(compare_any)
{
    custom_compare_any x{1};
    custom_compare_any y{2};
    EXPECT(1 == x);
    EXPECT(x != y);
    EXPECT(y > x);
    EXPECT(x == x);
    EXPECT(x < 2);
    EXPECT(y > 1);

    EXPECT_TOTALLY_ORDERED(x, x);
    EXPECT_TOTALLY_ORDERED(x, y);
    EXPECT_TOTALLY_ORDERED(x, 1);
}

struct custom_compare_equivalent : migraphx::totally_ordered<custom_compare_equivalent>,
                                   migraphx::equivalence<custom_compare_equivalent>
{
    int x;

    constexpr custom_compare_equivalent(int px) : x(px) {}

    constexpr bool operator<(const custom_compare_equivalent& rhs) const { return x < rhs.x; }

    template <class T>
    constexpr auto operator<(const T& rhs) const -> decltype(std::declval<int>() < rhs)
    {
        return x < rhs;
    }

    template <class T>
    constexpr auto operator>(const T& rhs) const -> decltype(std::declval<int>() > rhs)
    {
        return x > rhs;
    }

    template <class Stream>
    friend Stream& operator<<(Stream& os, const custom_compare_equivalent& self)
    {
        return os << self.x;
    }
};

TEST_CASE(compare_equivalent)
{
    custom_compare_equivalent x{1};
    custom_compare_equivalent y{2};
    EXPECT(1 == x);
    EXPECT(x != y);
    EXPECT(y > x);
    EXPECT(x == x);
    EXPECT(x < 2);
    EXPECT(y > 1);

    EXPECT_TOTALLY_ORDERED(x, x);
    EXPECT_TOTALLY_ORDERED(x, y);
    EXPECT_TOTALLY_ORDERED(x, 1);
}

struct custom_compare_adl : migraphx::totally_ordered<custom_compare_adl>
{
    int x;

    constexpr custom_compare_adl(int px) : x(px) {}

    friend constexpr bool operator==(const custom_compare_adl& lhs, const custom_compare_adl& rhs)
    {
        return lhs.x == rhs.x;
    }

    template <class T, MIGRAPHX_REQUIRES(not std::is_same<T, custom_compare_adl>{})>
    friend constexpr auto operator==(const custom_compare_adl& lhs,
                                     const T& rhs) -> decltype(std::declval<int>() == rhs)
    {
        return lhs.x == rhs;
    }

    friend constexpr bool operator<(const custom_compare_adl& lhs, const custom_compare_adl& rhs)
    {
        return lhs.x < rhs.x;
    }

    template <class T, MIGRAPHX_REQUIRES(not std::is_same<T, custom_compare_adl>{})>
    friend constexpr auto operator<(const custom_compare_adl& lhs,
                                    const T& rhs) -> decltype(std::declval<int>() < rhs)
    {
        return lhs.x < rhs;
    }

    template <class T, MIGRAPHX_REQUIRES(not std::is_same<T, custom_compare_adl>{})>
    friend constexpr auto operator>(const custom_compare_adl& lhs,
                                    const T& rhs) -> decltype(std::declval<int>() > rhs)
    {
        return lhs.x > rhs;
    }

    template <class Stream>
    friend Stream& operator<<(Stream& os, const custom_compare_adl& self)
    {
        return os << self.x;
    }
};

TEST_CASE(compare_adl)
{
    custom_compare_adl x{1};
    custom_compare_adl y{2};
    EXPECT(1 == x);
    EXPECT(x != y);
    EXPECT(x == x);
    EXPECT(y > x);
    EXPECT(x < 2);
    EXPECT(y > 1);

    EXPECT_TOTALLY_ORDERED(x, x);
    EXPECT_TOTALLY_ORDERED(x, y);
    EXPECT_TOTALLY_ORDERED(x, 1);
}

template <class T>
struct custom_compare_template1 : migraphx::totally_ordered<custom_compare_template1<T>>
{
    T x;

    constexpr custom_compare_template1(T px) : x(px) {}

    constexpr bool operator==(const custom_compare_template1& rhs) const { return x == rhs.x; }

    template <class U>
    constexpr auto operator==(const custom_compare_template1<U>& rhs) const
    {
        return x == rhs.x;
    }

    constexpr bool operator<(const custom_compare_template1& rhs) const { return x < rhs.x; }

    template <class U>
    constexpr auto operator<(const custom_compare_template1<U>& rhs) const
    {
        return x < rhs.x;
    }

    template <class Stream>
    friend Stream& operator<<(Stream& os, const custom_compare_template1& self)
    {
        return os << self.x;
    }
};

TEST_CASE(compare_template1)
{
    custom_compare_template1<int> x{1};
    custom_compare_template1<int> y{2};
    custom_compare_template1<short> z{2};
    EXPECT(x != y);
    EXPECT(x == x);
    EXPECT(y > x);
    EXPECT(x < y);
    EXPECT(z == z);
    EXPECT(z != x);
    EXPECT(z == y);
    EXPECT(z > x);
    EXPECT(x < z);

    EXPECT_TOTALLY_ORDERED(x, x);
    EXPECT_TOTALLY_ORDERED(x, y);
    EXPECT_TOTALLY_ORDERED(x, z);
    EXPECT_TOTALLY_ORDERED(y, z);
}

struct custom_addable : migraphx::arithmetic<custom_addable>,
                        migraphx::equality_comparable<custom_addable>
{
    int x;

    constexpr explicit custom_addable(int px) : x(px) {}

    constexpr auto operator+=(const custom_addable& rhs)
    {
        x += rhs.x;
        return *this;
    }

    constexpr auto operator+=(int rhs)
    {
        x += rhs;
        return *this;
    }

    template <class T>
    constexpr auto operator+=(const T& rhs) -> decltype(rhs.x, *this)
    {
        x += rhs.x;
        return *this;
    }

    constexpr bool operator==(const custom_addable& rhs) const { return x == rhs.x; }

    constexpr bool operator==(int rhs) const { return x == rhs; }

    template <class Stream>
    friend Stream& operator<<(Stream& os, const custom_addable& self)
    {
        return os << self.x;
    }
};

struct custom_addable_subtractable : migraphx::arithmetic<custom_addable_subtractable>,
                                     migraphx::equality_comparable<custom_addable_subtractable>
{
    int x;

    constexpr explicit custom_addable_subtractable(int px) : x(px) {}

    constexpr auto operator+=(const custom_addable_subtractable& rhs)
    {
        x += rhs.x;
        return *this;
    }

    constexpr auto operator+=(int rhs)
    {
        x += rhs;
        return *this;
    }

    constexpr auto operator-=(const custom_addable_subtractable& rhs)
    {
        x -= rhs.x;
        return *this;
    }

    constexpr auto operator-=(int rhs)
    {
        x -= rhs;
        return *this;
    }

    constexpr bool operator==(const custom_addable_subtractable& rhs) const { return x == rhs.x; }

    constexpr bool operator==(int rhs) const { return x == rhs; }

    template <class Stream>
    friend Stream& operator<<(Stream& os, const custom_addable_subtractable& self)
    {
        return os << self.x;
    }
};

struct custom_all_arithmetic : migraphx::arithmetic<custom_all_arithmetic>,
                               migraphx::equality_comparable<custom_all_arithmetic>
{
    int x;

    constexpr explicit custom_all_arithmetic(int px) : x(px) {}

    constexpr auto operator+=(const custom_all_arithmetic& rhs)
    {
        x += rhs.x;
        return *this;
    }

    constexpr auto operator+=(int rhs)
    {
        x += rhs;
        return *this;
    }

    constexpr auto operator-=(const custom_all_arithmetic& rhs)
    {
        x -= rhs.x;
        return *this;
    }

    constexpr auto operator-=(int rhs)
    {
        x -= rhs;
        return *this;
    }

    constexpr auto operator*=(const custom_all_arithmetic& rhs)
    {
        x *= rhs.x;
        return *this;
    }

    constexpr auto operator*=(int rhs)
    {
        x *= rhs;
        return *this;
    }

    constexpr auto operator/=(const custom_all_arithmetic& rhs)
    {
        x /= rhs.x;
        return *this;
    }

    constexpr auto operator/=(int rhs)
    {
        x /= rhs;
        return *this;
    }

    constexpr auto operator%=(const custom_all_arithmetic& rhs)
    {
        x %= rhs.x;
        return *this;
    }

    constexpr auto operator%=(int rhs)
    {
        x %= rhs;
        return *this;
    }

    template <class T>
    constexpr auto operator+=(const T& rhs) -> decltype(rhs.x, *this)
    {
        x += rhs.x;
        return *this;
    }

    template <class T>
    constexpr auto operator-=(const T& rhs) -> decltype(rhs.x, *this)
    {
        x -= rhs.x;
        return *this;
    }

    template <class T>
    constexpr auto operator*=(const T& rhs) -> decltype(rhs.x, *this)
    {
        x *= rhs.x;
        return *this;
    }

    template <class T>
    constexpr auto operator/=(const T& rhs) -> decltype(rhs.x, *this)
    {
        x /= rhs.x;
        return *this;
    }

    template <class T>
    constexpr auto operator%=(const T& rhs) -> decltype(rhs.x, *this)
    {
        x %= rhs.x;
        return *this;
    }

    constexpr bool operator==(const custom_all_arithmetic& rhs) const { return x == rhs.x; }

    constexpr bool operator==(int rhs) const { return x == rhs; }

    template <class Stream>
    friend Stream& operator<<(Stream& os, const custom_all_arithmetic& self)
    {
        return os << self.x;
    }
};

struct custom_all_arithmetic_adl : migraphx::arithmetic<custom_all_arithmetic_adl>,
                                   migraphx::equality_comparable<custom_all_arithmetic_adl>
{
    int x;

    constexpr explicit custom_all_arithmetic_adl(int px) : x(px) {}

    friend constexpr custom_all_arithmetic_adl& operator+=(custom_all_arithmetic_adl& lhs,
                                                           const custom_all_arithmetic_adl& rhs)
    {
        lhs.x += rhs.x;
        return lhs;
    }

    template <class T, MIGRAPHX_REQUIRES(not std::is_same<T, custom_all_arithmetic_adl>{})>
    friend constexpr auto operator+=(custom_all_arithmetic_adl& lhs, const T& rhs)
        -> decltype(std::declval<int&>() += rhs, lhs)
    {
        lhs.x += rhs;
        return lhs;
    }

    friend constexpr custom_all_arithmetic_adl& operator-=(custom_all_arithmetic_adl& lhs,
                                                           const custom_all_arithmetic_adl& rhs)
    {
        lhs.x -= rhs.x;
        return lhs;
    }

    template <class T, MIGRAPHX_REQUIRES(not std::is_same<T, custom_all_arithmetic_adl>{})>
    friend constexpr auto operator-=(custom_all_arithmetic_adl& lhs, const T& rhs)
        -> decltype(std::declval<int&>() -= rhs, lhs)
    {
        lhs.x -= rhs;
        return lhs;
    }

    friend constexpr custom_all_arithmetic_adl& operator*=(custom_all_arithmetic_adl& lhs,
                                                           const custom_all_arithmetic_adl& rhs)
    {
        lhs.x *= rhs.x;
        return lhs;
    }

    template <class T, MIGRAPHX_REQUIRES(not std::is_same<T, custom_all_arithmetic_adl>{})>
    friend constexpr auto operator*=(custom_all_arithmetic_adl& lhs, const T& rhs)
        -> decltype(std::declval<int&>() *= rhs, lhs)
    {
        lhs.x *= rhs;
        return lhs;
    }

    friend constexpr custom_all_arithmetic_adl& operator/=(custom_all_arithmetic_adl& lhs,
                                                           const custom_all_arithmetic_adl& rhs)
    {
        lhs.x /= rhs.x;
        return lhs;
    }

    template <class T, MIGRAPHX_REQUIRES(not std::is_same<T, custom_all_arithmetic_adl>{})>
    friend constexpr auto operator/=(custom_all_arithmetic_adl& lhs, const T& rhs)
        -> decltype(std::declval<int&>() /= rhs, lhs)
    {
        lhs.x /= rhs;
        return lhs;
    }

    friend constexpr custom_all_arithmetic_adl& operator%=(custom_all_arithmetic_adl& lhs,
                                                           const custom_all_arithmetic_adl& rhs)
    {
        lhs.x %= rhs.x;
        return lhs;
    }

    template <class T, MIGRAPHX_REQUIRES(not std::is_same<T, custom_all_arithmetic_adl>{})>
    friend constexpr auto operator%=(custom_all_arithmetic_adl& lhs, const T& rhs)
        -> decltype(std::declval<int&>() %= rhs, lhs)
    {
        lhs.x %= rhs;
        return lhs;
    }

    friend constexpr bool operator==(const custom_all_arithmetic_adl& lhs,
                                     const custom_all_arithmetic_adl& rhs)
    {
        return lhs.x == rhs.x;
    }

    template <class T, MIGRAPHX_REQUIRES(not std::is_same<T, custom_all_arithmetic_adl>{})>
    friend constexpr auto operator==(const custom_all_arithmetic_adl& lhs, const T& rhs)
        -> decltype(std::declval<int>() == rhs)
    {
        return lhs.x == rhs;
    }

    template <class Stream>
    friend Stream& operator<<(Stream& os, const custom_all_arithmetic_adl& self)
    {
        return os << self.x;
    }
};

template <class T>
struct custom_arithmetic_template : migraphx::arithmetic<custom_arithmetic_template<T>>,
                                    migraphx::equality_comparable<custom_arithmetic_template<T>>
{
    T x;

    constexpr explicit custom_arithmetic_template(T px) : x(px) {}

    constexpr auto operator+=(const custom_arithmetic_template& rhs)
    {
        x += rhs.x;
        return *this;
    }

    template <class U>
    constexpr auto operator+=(const custom_arithmetic_template<U>& rhs)
    {
        x += rhs.x;
        return *this;
    }

    constexpr auto operator-=(const custom_arithmetic_template& rhs)
    {
        x -= rhs.x;
        return *this;
    }

    template <class U>
    constexpr auto operator-=(const custom_arithmetic_template<U>& rhs)
    {
        x -= rhs.x;
        return *this;
    }

    constexpr auto operator*=(const custom_arithmetic_template& rhs)
    {
        x *= rhs.x;
        return *this;
    }

    template <class U>
    constexpr auto operator*=(const custom_arithmetic_template<U>& rhs)
    {
        x *= rhs.x;
        return *this;
    }

    constexpr bool operator==(const custom_arithmetic_template& rhs) const { return x == rhs.x; }

    template <class U>
    constexpr auto operator==(const custom_arithmetic_template<U>& rhs) const
    {
        return x == rhs.x;
    }

    constexpr bool operator<(const custom_arithmetic_template& rhs) const { return x < rhs.x; }

    template <class Stream>
    friend Stream& operator<<(Stream& os, const custom_arithmetic_template& self)
    {
        return os << self.x;
    }
};

template <class T>
struct custom_compare_template2 : migraphx::totally_ordered<custom_compare_template2<T>>
{
    T x;

    constexpr custom_compare_template2(T px) : x(px) {}

    template <class U>
    constexpr auto operator==(const custom_compare_template2<U>& rhs) const
    {
        return x == rhs.x;
    }

    template <class U>
    constexpr auto operator<(const custom_compare_template2<U>& rhs) const
    {
        return x < rhs.x;
    }

    template <class Stream>
    friend Stream& operator<<(Stream& os, const custom_compare_template2& self)
    {
        return os << self.x;
    }
};

TEST_CASE(compare_template2)
{
    custom_compare_template2<int> x{1};
    custom_compare_template2<int> y{2};
    custom_compare_template2<short> z{2};
    EXPECT(x != y);
    EXPECT(x == x);
    EXPECT(y > x);
    EXPECT(x < y);
    EXPECT(z == z);
    EXPECT(z != x);
    EXPECT(z == y);
    EXPECT(z > x);
    EXPECT(x < z);

    EXPECT_TOTALLY_ORDERED(x, x);
    EXPECT_TOTALLY_ORDERED(x, y);
    EXPECT_TOTALLY_ORDERED(x, z);
    EXPECT_TOTALLY_ORDERED(y, z);
}

TEST_CASE(arithmetic_addable_same_type)
{
    custom_addable a{3};
    custom_addable b{5};
    EXPECT(a + b == custom_addable{8});
    EXPECT(b + a == custom_addable{8});
    EXPECT(a + a == custom_addable{6});
}

TEST_CASE(arithmetic_addable_mixed)
{
    custom_addable a{3};
    EXPECT(a + 2 == custom_addable{5});
    EXPECT(2 + a == custom_addable{5});
    EXPECT(a + 0 == custom_addable{3});
    EXPECT(0 + a == custom_addable{3});
    EXPECT(a + 10 == custom_addable{13});
    EXPECT(10 + a == custom_addable{13});
}

TEST_CASE(arithmetic_addable_does_not_modify)
{
    custom_addable a{3};
    custom_addable b{5};
    auto c = a + b;
    EXPECT(a == custom_addable{3});
    EXPECT(b == custom_addable{5});
    EXPECT(c == custom_addable{8});
    auto d = a + 1;
    EXPECT(a == custom_addable{3});
    EXPECT(d == custom_addable{4});
    auto e = 1 + a;
    EXPECT(a == custom_addable{3});
    EXPECT(e == custom_addable{4});
}

TEST_CASE(arithmetic_addable_subtractable_add)
{
    custom_addable_subtractable a{10};
    custom_addable_subtractable b{3};
    EXPECT(a + b == custom_addable_subtractable{13});
    EXPECT(a + 5 == custom_addable_subtractable{15});
    EXPECT(5 + a == custom_addable_subtractable{15});
}

TEST_CASE(arithmetic_addable_subtractable_sub)
{
    custom_addable_subtractable a{10};
    custom_addable_subtractable b{3};
    EXPECT(a - b == custom_addable_subtractable{7});
    EXPECT(a - 4 == custom_addable_subtractable{6});
    EXPECT(b - a == custom_addable_subtractable{-7});
    EXPECT(b - 5 == custom_addable_subtractable{-2});
}

TEST_CASE(arithmetic_addable_subtractable_does_not_modify)
{
    custom_addable_subtractable a{10};
    custom_addable_subtractable b{3};
    auto c = a - b;
    EXPECT(a == custom_addable_subtractable{10});
    EXPECT(b == custom_addable_subtractable{3});
    EXPECT(c == custom_addable_subtractable{7});
}

TEST_CASE(arithmetic_all_add)
{
    custom_all_arithmetic a{7};
    custom_all_arithmetic b{3};
    EXPECT(a + b == custom_all_arithmetic{10});
    EXPECT(a + 2 == custom_all_arithmetic{9});
    EXPECT(2 + a == custom_all_arithmetic{9});
}

TEST_CASE(arithmetic_all_sub)
{
    custom_all_arithmetic a{7};
    custom_all_arithmetic b{3};
    EXPECT(a - b == custom_all_arithmetic{4});
    EXPECT(a - 2 == custom_all_arithmetic{5});
    EXPECT(10 - a == custom_all_arithmetic{3});
}

TEST_CASE(arithmetic_all_mul)
{
    custom_all_arithmetic a{7};
    custom_all_arithmetic b{3};
    EXPECT(a * b == custom_all_arithmetic{21});
    EXPECT(a * 2 == custom_all_arithmetic{14});
    EXPECT(2 * a == custom_all_arithmetic{14});
}

TEST_CASE(arithmetic_all_div)
{
    custom_all_arithmetic a{12};
    custom_all_arithmetic b{3};
    EXPECT(a / b == custom_all_arithmetic{4});
    EXPECT(a / 4 == custom_all_arithmetic{3});
    EXPECT(a / 2 == custom_all_arithmetic{6});
    EXPECT(20 / b == custom_all_arithmetic{6});
}

TEST_CASE(arithmetic_all_mod)
{
    custom_all_arithmetic a{10};
    custom_all_arithmetic b{3};
    EXPECT(a % b == custom_all_arithmetic{1});
    EXPECT(a % 4 == custom_all_arithmetic{2});
    EXPECT(a % 7 == custom_all_arithmetic{3});
    EXPECT(17 % b == custom_all_arithmetic{2});
}

TEST_CASE(arithmetic_all_does_not_modify)
{
    custom_all_arithmetic a{12};
    custom_all_arithmetic b{5};
    auto c1 = a + b;
    auto c2 = a - b;
    auto c3 = a * b;
    auto c4 = a / b;
    auto c5 = a % b;
    EXPECT(a == custom_all_arithmetic{12});
    EXPECT(b == custom_all_arithmetic{5});
    EXPECT(c1 == custom_all_arithmetic{17});
    EXPECT(c2 == custom_all_arithmetic{7});
    EXPECT(c3 == custom_all_arithmetic{60});
    EXPECT(c4 == custom_all_arithmetic{2});
    EXPECT(c5 == custom_all_arithmetic{2});
}

TEST_CASE(arithmetic_all_chain)
{
    custom_all_arithmetic a{2};
    custom_all_arithmetic b{3};
    custom_all_arithmetic c{4};
    EXPECT(a + b + c == custom_all_arithmetic{9});
    EXPECT(a * b + c == custom_all_arithmetic{10});
    EXPECT((a + b) * c == custom_all_arithmetic{20});
}

TEST_CASE(arithmetic_template_same_type)
{
    custom_arithmetic_template<int> a{3};
    custom_arithmetic_template<int> b{5};
    EXPECT(a + b == custom_arithmetic_template<int>{8});
    EXPECT(a - b == custom_arithmetic_template<int>{-2});
    EXPECT(a * b == custom_arithmetic_template<int>{15});
}

TEST_CASE(arithmetic_template_cross_type)
{
    custom_arithmetic_template<int> a{3};
    custom_arithmetic_template<short> b{5};
    EXPECT(a + b == custom_arithmetic_template<int>{8});
    EXPECT(b + a == custom_arithmetic_template<short>{8});
    EXPECT(a - b == custom_arithmetic_template<int>{-2});
    EXPECT(a * b == custom_arithmetic_template<int>{15});
}

TEST_CASE(arithmetic_template_does_not_modify)
{
    custom_arithmetic_template<int> a{4};
    custom_arithmetic_template<int> b{2};
    auto c = a + b;
    EXPECT(a == custom_arithmetic_template<int>{4});
    EXPECT(b == custom_arithmetic_template<int>{2});
    EXPECT(c == custom_arithmetic_template<int>{6});
}

TEST_CASE(arithmetic_adl_add)
{
    custom_all_arithmetic_adl a{7};
    custom_all_arithmetic_adl b{3};
    EXPECT(a + b == custom_all_arithmetic_adl{10});
    EXPECT(a + 2 == custom_all_arithmetic_adl{9});
    EXPECT(2 + a == custom_all_arithmetic_adl{9});
}

TEST_CASE(arithmetic_adl_sub)
{
    custom_all_arithmetic_adl a{7};
    custom_all_arithmetic_adl b{3};
    EXPECT(a - b == custom_all_arithmetic_adl{4});
    EXPECT(a - 2 == custom_all_arithmetic_adl{5});
    EXPECT(10 - a == custom_all_arithmetic_adl{3});
}

TEST_CASE(arithmetic_adl_mul)
{
    custom_all_arithmetic_adl a{7};
    custom_all_arithmetic_adl b{3};
    EXPECT(a * b == custom_all_arithmetic_adl{21});
    EXPECT(a * 2 == custom_all_arithmetic_adl{14});
    EXPECT(2 * a == custom_all_arithmetic_adl{14});
}

TEST_CASE(arithmetic_adl_div)
{
    custom_all_arithmetic_adl a{12};
    custom_all_arithmetic_adl b{3};
    EXPECT(a / b == custom_all_arithmetic_adl{4});
    EXPECT(a / 4 == custom_all_arithmetic_adl{3});
    EXPECT(20 / b == custom_all_arithmetic_adl{6});
}

TEST_CASE(arithmetic_adl_mod)
{
    custom_all_arithmetic_adl a{10};
    custom_all_arithmetic_adl b{3};
    EXPECT(a % b == custom_all_arithmetic_adl{1});
    EXPECT(a % 4 == custom_all_arithmetic_adl{2});
    EXPECT(17 % b == custom_all_arithmetic_adl{2});
}

TEST_CASE(arithmetic_adl_does_not_modify)
{
    custom_all_arithmetic_adl a{12};
    custom_all_arithmetic_adl b{5};
    auto c1 = a + b;
    auto c2 = a - b;
    auto c3 = a * b;
    auto c4 = a / b;
    auto c5 = a % b;
    EXPECT(a == custom_all_arithmetic_adl{12});
    EXPECT(b == custom_all_arithmetic_adl{5});
    EXPECT(c1 == custom_all_arithmetic_adl{17});
    EXPECT(c2 == custom_all_arithmetic_adl{7});
    EXPECT(c3 == custom_all_arithmetic_adl{60});
    EXPECT(c4 == custom_all_arithmetic_adl{2});
    EXPECT(c5 == custom_all_arithmetic_adl{2});
}

TEST_CASE(arithmetic_adl_chain)
{
    custom_all_arithmetic_adl a{2};
    custom_all_arithmetic_adl b{3};
    custom_all_arithmetic_adl c{4};
    EXPECT(a + b + c == custom_all_arithmetic_adl{9});
    EXPECT(a * b + c == custom_all_arithmetic_adl{10});
    EXPECT((a + b) * c == custom_all_arithmetic_adl{20});
}

TEST_CASE(arithmetic_cross_class_add)
{
    custom_all_arithmetic a{1};
    custom_addable b{2};
    EXPECT(a + b == custom_all_arithmetic{3});
    EXPECT(b + a == custom_addable{3});
}

TEST_CASE(arithmetic_cross_class_sub)
{
    custom_all_arithmetic a{10};
    custom_addable b{3};
    EXPECT(a - b == custom_all_arithmetic{7});
}

TEST_CASE(arithmetic_cross_class_mul)
{
    custom_all_arithmetic a{4};
    custom_addable b{3};
    EXPECT(a * b == custom_all_arithmetic{12});
}

TEST_CASE(arithmetic_cross_class_div)
{
    custom_all_arithmetic a{12};
    custom_addable b{3};
    EXPECT(a / b == custom_all_arithmetic{4});
}

TEST_CASE(arithmetic_cross_class_mod)
{
    custom_all_arithmetic a{10};
    custom_addable b{3};
    EXPECT(a % b == custom_all_arithmetic{1});
}

TEST_CASE(arithmetic_cross_class_does_not_modify)
{
    custom_all_arithmetic a{10};
    custom_addable b{3};
    auto c = a + b;
    EXPECT(a == custom_all_arithmetic{10});
    EXPECT(b == custom_addable{3});
    EXPECT(c == custom_all_arithmetic{13});
    auto d = b + a;
    EXPECT(a == custom_all_arithmetic{10});
    EXPECT(b == custom_addable{3});
    EXPECT(d == custom_addable{13});
}

struct custom_bitwise_and_or : migraphx::bitwise<custom_bitwise_and_or>,
                               migraphx::equality_comparable<custom_bitwise_and_or>
{
    int x;

    constexpr explicit custom_bitwise_and_or(int px) : x(px) {}

    constexpr auto operator&=(const custom_bitwise_and_or& rhs)
    {
        x &= rhs.x;
        return *this;
    }

    constexpr auto operator&=(int rhs)
    {
        x &= rhs;
        return *this;
    }

    constexpr auto operator|=(const custom_bitwise_and_or& rhs)
    {
        x |= rhs.x;
        return *this;
    }

    constexpr auto operator|=(int rhs)
    {
        x |= rhs;
        return *this;
    }

    constexpr bool operator==(const custom_bitwise_and_or& rhs) const { return x == rhs.x; }

    constexpr bool operator==(int rhs) const { return x == rhs; }

    template <class Stream>
    friend Stream& operator<<(Stream& os, const custom_bitwise_and_or& self)
    {
        return os << self.x;
    }
};

struct custom_all_bitwise : migraphx::bitwise<custom_all_bitwise>,
                            migraphx::equality_comparable<custom_all_bitwise>
{
    int x;

    constexpr explicit custom_all_bitwise(int px) : x(px) {}

    constexpr auto operator&=(const custom_all_bitwise& rhs)
    {
        x &= rhs.x;
        return *this;
    }

    constexpr auto operator&=(int rhs)
    {
        x &= rhs;
        return *this;
    }

    constexpr auto operator|=(const custom_all_bitwise& rhs)
    {
        x |= rhs.x;
        return *this;
    }

    constexpr auto operator|=(int rhs)
    {
        x |= rhs;
        return *this;
    }

    constexpr auto operator^=(const custom_all_bitwise& rhs)
    {
        x ^= rhs.x;
        return *this;
    }

    constexpr auto operator^=(int rhs)
    {
        x ^= rhs;
        return *this;
    }

    constexpr auto operator<<=(const custom_all_bitwise& rhs)
    {
        x <<= rhs.x;
        return *this;
    }

    constexpr auto operator<<=(int rhs)
    {
        x <<= rhs;
        return *this;
    }

    constexpr auto operator>>=(const custom_all_bitwise& rhs)
    {
        x >>= rhs.x;
        return *this;
    }

    constexpr auto operator>>=(int rhs)
    {
        x >>= rhs;
        return *this;
    }

    constexpr bool operator==(const custom_all_bitwise& rhs) const { return x == rhs.x; }

    constexpr bool operator==(int rhs) const { return x == rhs; }

    template <class Stream>
    friend Stream& operator<<(Stream& os, const custom_all_bitwise& self)
    {
        return os << self.x;
    }
};

struct custom_all_bitwise_adl : migraphx::bitwise<custom_all_bitwise_adl>,
                                migraphx::equality_comparable<custom_all_bitwise_adl>
{
    int x;

    constexpr explicit custom_all_bitwise_adl(int px) : x(px) {}

    friend constexpr custom_all_bitwise_adl& operator&=(custom_all_bitwise_adl& lhs,
                                                        const custom_all_bitwise_adl& rhs)
    {
        lhs.x &= rhs.x;
        return lhs;
    }

    template <class T, MIGRAPHX_REQUIRES(not std::is_same<T, custom_all_bitwise_adl>{})>
    friend constexpr auto operator&=(custom_all_bitwise_adl& lhs, const T& rhs)
        -> decltype(std::declval<int&>() &= rhs, lhs)
    {
        lhs.x &= rhs;
        return lhs;
    }

    friend constexpr custom_all_bitwise_adl& operator|=(custom_all_bitwise_adl& lhs,
                                                        const custom_all_bitwise_adl& rhs)
    {
        lhs.x |= rhs.x;
        return lhs;
    }

    template <class T, MIGRAPHX_REQUIRES(not std::is_same<T, custom_all_bitwise_adl>{})>
    friend constexpr auto operator|=(custom_all_bitwise_adl& lhs, const T& rhs)
        -> decltype(std::declval<int&>() |= rhs, lhs)
    {
        lhs.x |= rhs;
        return lhs;
    }

    friend constexpr custom_all_bitwise_adl& operator^=(custom_all_bitwise_adl& lhs,
                                                        const custom_all_bitwise_adl& rhs)
    {
        lhs.x ^= rhs.x;
        return lhs;
    }

    template <class T, MIGRAPHX_REQUIRES(not std::is_same<T, custom_all_bitwise_adl>{})>
    friend constexpr auto operator^=(custom_all_bitwise_adl& lhs, const T& rhs)
        -> decltype(std::declval<int&>() ^= rhs, lhs)
    {
        lhs.x ^= rhs;
        return lhs;
    }

    friend constexpr custom_all_bitwise_adl& operator<<=(custom_all_bitwise_adl& lhs,
                                                         const custom_all_bitwise_adl& rhs)
    {
        lhs.x <<= rhs.x;
        return lhs;
    }

    template <class T, MIGRAPHX_REQUIRES(not std::is_same<T, custom_all_bitwise_adl>{})>
    friend constexpr auto operator<<=(custom_all_bitwise_adl& lhs, const T& rhs)
        -> decltype(std::declval<int&>() <<= rhs, lhs)
    {
        lhs.x <<= rhs;
        return lhs;
    }

    friend constexpr custom_all_bitwise_adl& operator>>=(custom_all_bitwise_adl& lhs,
                                                         const custom_all_bitwise_adl& rhs)
    {
        lhs.x >>= rhs.x;
        return lhs;
    }

    template <class T, MIGRAPHX_REQUIRES(not std::is_same<T, custom_all_bitwise_adl>{})>
    friend constexpr auto operator>>=(custom_all_bitwise_adl& lhs, const T& rhs)
        -> decltype(std::declval<int&>() >>= rhs, lhs)
    {
        lhs.x >>= rhs;
        return lhs;
    }

    friend constexpr bool operator==(const custom_all_bitwise_adl& lhs,
                                     const custom_all_bitwise_adl& rhs)
    {
        return lhs.x == rhs.x;
    }

    template <class T, MIGRAPHX_REQUIRES(not std::is_same<T, custom_all_bitwise_adl>{})>
    friend constexpr auto operator==(const custom_all_bitwise_adl& lhs, const T& rhs)
        -> decltype(std::declval<int>() == rhs)
    {
        return lhs.x == rhs;
    }

    template <class Stream>
    friend Stream& operator<<(Stream& os, const custom_all_bitwise_adl& self)
    {
        return os << self.x;
    }
};

TEST_CASE(bitwise_and_or_same_type)
{
    custom_bitwise_and_or a{0b1100};
    custom_bitwise_and_or b{0b1010};
    EXPECT((a & b) == custom_bitwise_and_or{0b1000});
    EXPECT((a | b) == custom_bitwise_and_or{0b1110});
}

TEST_CASE(bitwise_and_or_mixed)
{
    custom_bitwise_and_or a{0b1100};
    EXPECT((a & 0b1010) == custom_bitwise_and_or{0b1000});
    EXPECT((0b1010 & a) == custom_bitwise_and_or{0b1000});
    EXPECT((a | 0b0011) == custom_bitwise_and_or{0b1111});
    EXPECT((0b0011 | a) == custom_bitwise_and_or{0b1111});
}

TEST_CASE(bitwise_and_or_does_not_modify)
{
    custom_bitwise_and_or a{0b1100};
    custom_bitwise_and_or b{0b1010};
    auto c = a & b;
    EXPECT(a == custom_bitwise_and_or{0b1100});
    EXPECT(b == custom_bitwise_and_or{0b1010});
    EXPECT(c == custom_bitwise_and_or{0b1000});
}

TEST_CASE(bitwise_all_and)
{
    custom_all_bitwise a{0b1100};
    custom_all_bitwise b{0b1010};
    EXPECT((a & b) == custom_all_bitwise{0b1000});
    EXPECT((a & 0b1010) == custom_all_bitwise{0b1000});
    EXPECT((0b1010 & a) == custom_all_bitwise{0b1000});
}

TEST_CASE(bitwise_all_or)
{
    custom_all_bitwise a{0b1100};
    custom_all_bitwise b{0b0011};
    EXPECT((a | b) == custom_all_bitwise{0b1111});
    EXPECT((a | 0b0011) == custom_all_bitwise{0b1111});
    EXPECT((0b0011 | a) == custom_all_bitwise{0b1111});
}

TEST_CASE(bitwise_all_xor)
{
    custom_all_bitwise a{0b1100};
    custom_all_bitwise b{0b1010};
    EXPECT((a ^ b) == custom_all_bitwise{0b0110});
    EXPECT((a ^ 0b1010) == custom_all_bitwise{0b0110});
    EXPECT((0b1010 ^ a) == custom_all_bitwise{0b0110});
}

TEST_CASE(bitwise_all_shl)
{
    custom_all_bitwise a{0b0011};
    custom_all_bitwise b{2};
    EXPECT((a << b) == custom_all_bitwise{0b1100});
    EXPECT((a << 2) == custom_all_bitwise{0b1100});
    EXPECT((1 << b) == custom_all_bitwise{4});
}

TEST_CASE(bitwise_all_shr)
{
    custom_all_bitwise a{0b1100};
    custom_all_bitwise b{2};
    EXPECT((a >> b) == custom_all_bitwise{0b0011});
    EXPECT((a >> 2) == custom_all_bitwise{0b0011});
    EXPECT((16 >> b) == custom_all_bitwise{4});
}

TEST_CASE(bitwise_all_does_not_modify)
{
    custom_all_bitwise a{0b1100};
    custom_all_bitwise b{0b1010};
    auto c1 = a & b;
    auto c2 = a | b;
    auto c3 = a ^ b;
    EXPECT(a == custom_all_bitwise{0b1100});
    EXPECT(b == custom_all_bitwise{0b1010});
    EXPECT(c1 == custom_all_bitwise{0b1000});
    EXPECT(c2 == custom_all_bitwise{0b1110});
    EXPECT(c3 == custom_all_bitwise{0b0110});
}

TEST_CASE(bitwise_all_chain)
{
    custom_all_bitwise a{0b1100};
    custom_all_bitwise b{0b1010};
    custom_all_bitwise c{0b0110};
    EXPECT((a & b | c) == custom_all_bitwise{0b1110});
    EXPECT(((a | b) ^ c) == custom_all_bitwise{0b1000});
}

TEST_CASE(bitwise_adl_and)
{
    custom_all_bitwise_adl a{0b1100};
    custom_all_bitwise_adl b{0b1010};
    EXPECT((a & b) == custom_all_bitwise_adl{0b1000});
    EXPECT((a & 0b1010) == custom_all_bitwise_adl{0b1000});
    EXPECT((0b1010 & a) == custom_all_bitwise_adl{0b1000});
}

TEST_CASE(bitwise_adl_or)
{
    custom_all_bitwise_adl a{0b1100};
    custom_all_bitwise_adl b{0b0011};
    EXPECT((a | b) == custom_all_bitwise_adl{0b1111});
    EXPECT((a | 0b0011) == custom_all_bitwise_adl{0b1111});
    EXPECT((0b0011 | a) == custom_all_bitwise_adl{0b1111});
}

TEST_CASE(bitwise_adl_xor)
{
    custom_all_bitwise_adl a{0b1100};
    custom_all_bitwise_adl b{0b1010};
    EXPECT((a ^ b) == custom_all_bitwise_adl{0b0110});
    EXPECT((a ^ 0b1010) == custom_all_bitwise_adl{0b0110});
    EXPECT((0b1010 ^ a) == custom_all_bitwise_adl{0b0110});
}

TEST_CASE(bitwise_adl_shl)
{
    custom_all_bitwise_adl a{0b0011};
    custom_all_bitwise_adl b{2};
    EXPECT((a << b) == custom_all_bitwise_adl{0b1100});
    EXPECT((a << 2) == custom_all_bitwise_adl{0b1100});
    EXPECT((1 << b) == custom_all_bitwise_adl{4});
}

TEST_CASE(bitwise_adl_shr)
{
    custom_all_bitwise_adl a{0b1100};
    custom_all_bitwise_adl b{2};
    EXPECT((a >> b) == custom_all_bitwise_adl{0b0011});
    EXPECT((a >> 2) == custom_all_bitwise_adl{0b0011});
    EXPECT((16 >> b) == custom_all_bitwise_adl{4});
}

TEST_CASE(bitwise_adl_does_not_modify)
{
    custom_all_bitwise_adl a{0b1100};
    custom_all_bitwise_adl b{0b1010};
    auto c1 = a & b;
    auto c2 = a | b;
    auto c3 = a ^ b;
    EXPECT(a == custom_all_bitwise_adl{0b1100});
    EXPECT(b == custom_all_bitwise_adl{0b1010});
    EXPECT(c1 == custom_all_bitwise_adl{0b1000});
    EXPECT(c2 == custom_all_bitwise_adl{0b1110});
    EXPECT(c3 == custom_all_bitwise_adl{0b0110});
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
